"""Student model for MP1.

The contribution is a small GPT variant with RMSNorm, rotary position
embeddings and a SwiGLU feed-forward block. The public interface matches the
baseline: forward(ids) returns logits and predict_log_probs(ids) returns
normalized causal log probabilities.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class Rotary(nn.Module):
    def __init__(self, dim, context, base=10000.0):
        super().__init__()
        inv_freq = base ** (-torch.arange(0, dim, 2).float() / dim)
        positions = torch.arange(context).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q, k):
        cos, sin = self.cos[:, :, : q.shape[-2]], self.sin[:, :, : q.shape[-2]]
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class StudentBlock(nn.Module):
    def __init__(self, width, heads, context, mlp_hidden, use_rope=True):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.use_rope = use_rope
        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.rope = Rotary(self.head_dim, context) if use_rope else None
        self.mlp = SwiGLU(width, mlp_hidden)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        if self.use_rope:
            q, k = self.rope(q, k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config["context"]
        self.vocab = config["vocab"]
        self.neural_weight = config.get("student_neural_weight", 1.0)
        width = config.get("student_width", config["width"])
        heads = config.get("student_heads", config["heads"])
        depth = config.get("student_depth", config["depth"])
        if width % heads != 0 or (width // heads) % 2 != 0:
            raise ValueError("width // heads must be even for RoPE.")
        mlp_hidden = config.get("student_mlp_hidden", int(math.ceil((8 * width / 3) / 32) * 32))
        use_rope = config.get("student_use_rope", True)

        self.token = nn.Embedding(self.vocab, width)
        self.blocks = nn.ModuleList(
            StudentBlock(width, heads, self.context, mlp_hidden, use_rope) for _ in range(depth)
        )
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, self.vocab, bias=False)
        if self.neural_weight < 1.0:
            self.register_buffer("bigram_log_probs", torch.empty(self.vocab, self.vocab))
            trigram_contexts = config.get("student_trigram_contexts", 0)
            trigram_topk = config.get("student_trigram_topk", 0)
            if trigram_contexts and trigram_topk:
                self.register_buffer("trigram_keys", torch.empty(trigram_contexts, dtype=torch.long))
                self.register_buffer("trigram_tokens", torch.empty(trigram_contexts, trigram_topk, dtype=torch.long))
                self.register_buffer("trigram_log_probs", torch.empty(trigram_contexts, trigram_topk))
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        neural = F.log_softmax(self(ids).float(), dim=-1)
        if self.neural_weight >= 1.0:
            return neural
        bigram = self.bigram_log_probs[ids].to(neural.device, dtype=neural.dtype)
        if hasattr(self, "trigram_keys"):
            flat_ids = ids.reshape(-1)
            previous = torch.zeros_like(ids)
            previous[:, 1:] = ids[:, :-1]
            codes = (previous * self.vocab + ids).reshape(-1)
            valid_position = torch.ones_like(ids, dtype=torch.bool).reshape(-1)
            valid_position[:: ids.shape[1]] = False
            keys = self.trigram_keys.to(ids.device)
            lookup = torch.searchsorted(keys, codes)
            in_range = lookup < keys.numel()
            found = in_range & valid_position
            found_indices = torch.zeros_like(lookup)
            found_indices[in_range] = lookup[in_range]
            found = found & (keys[found_indices] == codes)
            if found.any():
                ngram = bigram.reshape(-1, self.vocab)
                rows = found.nonzero(as_tuple=False).flatten()
                replacement = torch.full(
                    (rows.numel(), self.vocab),
                    -torch.inf,
                    device=neural.device,
                    dtype=neural.dtype,
                )
                trigram_tokens = self.trigram_tokens.to(ids.device)[found_indices[rows]]
                trigram_log_probs = self.trigram_log_probs.to(neural.device, dtype=neural.dtype)[found_indices[rows]]
                replacement.scatter_(1, trigram_tokens, trigram_log_probs)
                ngram[rows] = replacement
                bigram = ngram.view_as(bigram)
        return torch.logaddexp(
            neural + math.log(self.neural_weight),
            bigram + math.log1p(-self.neural_weight),
        )


def build_model(config):
    return StudentGPT(config)
