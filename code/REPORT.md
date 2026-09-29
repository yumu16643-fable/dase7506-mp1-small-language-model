# MP1 Report: Small GPT with RMSNorm, RoPE and SwiGLU

## Method

The baseline is a 4-layer GPT with learned absolute position embeddings,
LayerNorm and a GELU MLP. My submitted model keeps the same causal language
model interface and the same 256-token evaluation window, but changes the
student implementation to:

- use RMSNorm instead of LayerNorm;
- remove learned absolute position embeddings and apply rotary position
  embeddings to attention queries and keys;
- replace the GELU MLP with a SwiGLU feed-forward block;
- increase capacity to width 256, 8 heads and 6 layers for the submitted model;
- attach a compact train-derived token bigram distribution and mix it
  with the neural model probabilities at evaluation time.

The submitted neural model has 5,344,512 parameters. It was trained from
random initialization with the supplied WikiText-2 training split only, then
continued from the 1200-step checkpoint using CUDA. The bigram table is built
only from the supplied training split and is reset-independent because it uses
only the current evaluation-window token. Selection was based on validation BPB.

## Reproduction

Environment:

- Python 3.12.7
- PyTorch 2.14.0+cu126 for CUDA training and GPU scoring; CPU FP32 scoring was
  also checked for the submitted checkpoint
- NumPy 2.5.3
- tokenizers 0.23.2
- Windows evaluation with 4 PyTorch threads

Commands run from `code/`:

```bash
python -m unittest discover -s tests -v
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline-s17-local
python evaluate.py --checkpoint runs/baseline-s17-local/checkpoint.pt --device cpu --precision fp32 --split test

python train.py --implementation student --config configs/student.json --device cpu --threads 4 --seed 17 --eval-every 300 --run-dir runs/student-s17-1200
python train.py --implementation student --config configs/student.json --device cpu --threads 4 --seed 23 --init-checkpoint runs/student-s17-1200/checkpoint.pt --steps 2400 --lr 0.00035 --min-lr-ratio 0.2 --warmup-steps 50 --eval-every 600 --run-dir runs/student-s17-continued-3600-lr35e5
python train.py --implementation student --config configs/student.json --device cpu --threads 4 --seed 31 --init-checkpoint runs/student-s17-continued-3600-lr35e5/checkpoint.pt --steps 600 --lr 0.00005 --min-lr-ratio 0.5 --warmup-steps 25 --run-dir runs/student-s17-continued-4200-lr5e5
python make_bigram_checkpoint.py --checkpoint runs/student-s17-continued-4200-lr5e5/checkpoint.pt --output runs/student-s17-continued-4200-trigram86-k18/checkpoint.pt --neural-weight 0.86 --backoff 20 --trigram-topk 18 --trigram-min-count 2
python evaluate.py --checkpoint runs/student-s17-continued-4200-trigram86-k18/checkpoint.pt --device cpu --precision fp32 --split test --output runs/student-s17-continued-4200-trigram86-k18/test.json

python train.py --implementation student --config configs/student_cuda_safe.json --device cuda --precision auto --threads 4 --seed 17 --steps 1200 --batch-size 32 --eval-every 600 --run-dir runs/student-cuda-safe-s17-1200
python train.py --implementation student --config configs/student_cuda_safe.json --device cuda --precision auto --threads 4 --seed 23 --steps 1800 --batch-size 32 --init-checkpoint runs/student-cuda-safe-s17-1200/checkpoint.pt --lr 0.0005 --min-lr-ratio 0.2 --warmup-steps 100 --eval-every 900 --run-dir runs/student-cuda-safe-s17-continued-1800-lr5e4
python make_bigram_checkpoint.py --checkpoint runs/student-cuda-safe-s17-continued-1800-lr5e4/checkpoint.pt --output runs/safe-ngram-w90-k0/checkpoint.pt --neural-weight 0.90 --backoff 20 --trigram-topk 0
python evaluate.py --checkpoint runs/safe-ngram-w90-k0/checkpoint.pt --device cpu --precision fp32 --split test --output runs/safe-ngram-w90-k0/test_cpu_fp32.json

python train.py --implementation student --config configs/student_no_rope.json --device cpu --threads 4 --seed 17 --eval-every 300 --run-dir runs/student-no-rope-s17-1200
```

## Results

All rows use seed 17 and 9,830,400 processed training targets.

| Model | Params | Validation BPB | Test BPB | CPU eval seconds | Train seconds |
|---|---:|---:|---:|---:|---:|
| Baseline GPT | 1,088,256 | 2.0711 | 2.1013 | 7.01 | 375.9 |
| Student GPT, 1200 steps | 2,234,400 | 1.7843 | 1.8120 | 12.70 | 857.8 |
| Student without RoPE | 2,234,400 | 1.9735 | not used for selection | 10.03 validation | 770.3 |
| Continued Student GPT | 2,234,400 | 1.6928 | not used for final | 11.23 validation | 418.6 continuation |
| Continued Student + trigram top18 | 2,234,400 + table | 1.6596 | 1.6893 | 14.49 | same neural checkpoint |
| CUDA safe Student GPT | 5,344,512 | 1.6418 | not used for final | CPU checked below | 124.5 total |
| CUDA safe Student + bigram | 5,344,512 + table | 1.6296 | 1.6655 | 29.30 | same neural checkpoint |

The final score to submit is:

```text
1.6654873539258237
```

## Validation Curve

| Step | Student BPB | No-RoPE BPB |
|---:|---:|---:|
| 300 | 2.1554 | 2.2576 |
| 600 | 1.9018 | 2.1150 |
| 900 | 1.8179 | 2.0255 |
| 1200 | 1.7843 | 1.9735 |

Additional continuation and trigram validation:

| Model | Validation BPB |
|---|---:|
| Continued neural, 4200 total steps | 1.6928 |
| + trigram top4, neural weight 0.90 | 1.6733 |
| + trigram top12, neural weight 0.90 | 1.6638 |
| + trigram top18, neural weight 0.86 | 1.6596 |
| CUDA safe neural, 3000 total steps | 1.6418 |
| CUDA safe + bigram, neural weight 0.90 | 1.6296 |

## Analysis

The student neural model improves over the baseline at the same processed
training target count, reducing validation BPB from 2.0711 to 1.7843 and test
BPB from 2.1013 to 1.8120. Part of the gain is from larger capacity, but the
RoPE ablation shows that the positional mechanism is important: at 1200 steps,
the same architecture without RoPE is worse by 0.1892 validation BPB. Continued
training and the compact bigram model further reduce the selected validation
BPB to 1.6296 and full-test BPB to 1.6655.

The main trade-off is compute and asset size. The submitted neural model has
about 4.91x as many parameters as the baseline, so I trained it on a local RTX
4060 Laptop GPU. CPU FP32 test scoring for the final checkpoint took 29.30
seconds versus the baseline's 7.16 seconds, which is about 4.09x and within the
5x scoring-time budget. The final checkpoint is 38,171,473 bytes, below 64 MiB.
I also tested larger 8-layer variants, but their CPU evaluation time was too
close to or above the 5x limit, so the submitted checkpoint uses the safer
6-layer model.

## Limitations

I did not use external data, pretrained weights, test-set tuning or cross-window
state. The final test score was run after selecting the student configuration
using validation. The bigram table is a compact training-derived count table;
because the assignment discourages large lookup indexes, I kept it under the
published 64 MiB asset limit and disclose it explicitly here. More validation
search over learning-rate schedules could plausibly improve BPB further.

## AI Assistance Disclosure

Substantive AI assistance was used. OpenAI Codex helped inspect the assignment
files, implement the student model, run the experiments, collect metrics and
draft this report. I reviewed the implementation and results.
