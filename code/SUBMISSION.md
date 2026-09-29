# Submission Notes

Final full-test BPB:

```text
1.6654873539258237
```

Final checkpoint:

```text
runs/safe-ngram-w90-k0/checkpoint.pt
```

Final evaluation JSON:

```text
runs/safe-ngram-w90-k0/test_cpu_fp32.json
```

Reproduce the submitted score from `code/`:

```bash
python evaluate.py --checkpoint runs/safe-ngram-w90-k0/checkpoint.pt --device cpu --precision fp32 --split test --output reproduced-test.json
```

For the course website, submit:

- student ID: your student ID
- full-test BPB: `1.6654873539258237`
- code link: immutable GitHub repository or release link containing this code and `REPORT.md`
- checkpoint bundle link: downloadable archive containing the final checkpoint and any required assets

The final website submission must include both links by the end of 30 September
2026 UTC+8.
