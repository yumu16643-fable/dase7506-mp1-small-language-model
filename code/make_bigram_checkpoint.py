"""Attach a compact train-derived bigram model to a student checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from common import PROTOCOL, ROOT, load_data, make_model, sha


def build_bigram_log_probs(tokens, vocab, backoff):
    counts = torch.zeros(vocab, vocab, dtype=torch.float32)
    current, nxt = tokens[:-1], tokens[1:]
    ones = torch.ones_like(current, dtype=torch.float32)
    counts.index_put_((current, nxt), ones, accumulate=True)
    unigram = torch.bincount(nxt, minlength=vocab).float()
    unigram = (unigram + 1.0) / (unigram.sum() + vocab)
    row_counts = counts.sum(dim=1, keepdim=True)
    probs = (counts + backoff * unigram.unsqueeze(0)) / (row_counts + backoff)
    return probs.log()


def build_trigram_topk(tokens, vocab, topk, min_count):
    codes = (tokens[:-2].long() * vocab + tokens[1:-1].long())
    combined = codes * vocab + tokens[2:].long()
    unique, counts = torch.unique(combined, return_counts=True)
    contexts = torch.div(unique, vocab, rounding_mode="floor")
    next_tokens = unique.remainder(vocab)
    order = torch.argsort(contexts * (tokens.numel() + 1) - counts, stable=True)
    contexts, next_tokens, counts = contexts[order], next_tokens[order], counts[order].float()

    key_rows, token_rows, prob_rows = [], [], []
    start = 0
    while start < contexts.numel():
        end = start + 1
        while end < contexts.numel() and contexts[end] == contexts[start]:
            end += 1
        group_counts = counts[start:end]
        if group_counts.sum().item() >= min_count:
            take = min(topk, end - start)
            selected_tokens = next_tokens[start:start + take]
            selected_counts = group_counts[:take]
            probs = selected_counts / selected_counts.sum()
            if take < topk:
                selected_tokens = torch.cat([selected_tokens, selected_tokens.new_zeros(topk - take)])
                probs = torch.cat([probs, probs.new_full((topk - take,), 1e-12)])
                probs = probs / probs.sum()
            key_rows.append(contexts[start])
            token_rows.append(selected_tokens)
            prob_rows.append(probs.log())
        start = end

    if not key_rows:
        return torch.empty(0, dtype=torch.long), torch.empty(0, topk, dtype=torch.long), torch.empty(0, topk)
    return torch.stack(key_rows).long(), torch.stack(token_rows).long(), torch.stack(prob_rows).float()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--neural-weight", type=float, default=0.9)
    p.add_argument("--backoff", type=float, default=20.0)
    p.add_argument("--trigram-topk", type=int, default=0)
    p.add_argument("--trigram-min-count", type=int, default=2)
    args = p.parse_args()
    if not 0.0 < args.neural_weight < 1.0:
        p.error("--neural-weight must be between 0 and 1.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = dict(checkpoint["config"])
    data = load_data()
    trigram_keys = trigram_tokens = trigram_log_probs = None
    if args.trigram_topk:
        trigram_keys, trigram_tokens, trigram_log_probs = build_trigram_topk(
            data["train"][0], config["vocab"], args.trigram_topk, args.trigram_min_count
        )
        config["student_trigram_contexts"] = int(trigram_keys.numel())
        config["student_trigram_topk"] = args.trigram_topk
        config["student_trigram_min_count"] = args.trigram_min_count
    config["student_neural_weight"] = args.neural_weight
    config["student_bigram_backoff"] = args.backoff

    model, implementation_sha = make_model(checkpoint["implementation"], config, torch.device("cpu"))
    model_state = model.state_dict()
    for key, value in checkpoint["model"].items():
        model_state[key].copy_(value)
    model_state["bigram_log_probs"].copy_(
        build_bigram_log_probs(data["train"][0], config["vocab"], args.backoff)
    )
    if args.trigram_topk:
        model_state["trigram_keys"].copy_(trigram_keys)
        model_state["trigram_tokens"].copy_(trigram_tokens)
        model_state["trigram_log_probs"].copy_(trigram_log_probs)
    model.load_state_dict(model_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "protocol": PROTOCOL,
        "implementation": checkpoint["implementation"],
        "config": config,
        "model": model.state_dict(),
        "seed": checkpoint.get("seed"),
        "train_tokens": checkpoint.get("train_tokens", 0),
        "source_checkpoint_sha256": sha(args.checkpoint),
        "implementation_sha256": implementation_sha,
    }, args.output)
    print(json.dumps({
        "output": str(args.output),
        "checkpoint_sha256": sha(args.output),
        "source_checkpoint_sha256": sha(args.checkpoint),
        "neural_weight": args.neural_weight,
        "backoff": args.backoff,
        "trigram_contexts": int(config.get("student_trigram_contexts", 0)),
        "trigram_topk": args.trigram_topk,
    }, indent=2))


if __name__ == "__main__":
    main()
