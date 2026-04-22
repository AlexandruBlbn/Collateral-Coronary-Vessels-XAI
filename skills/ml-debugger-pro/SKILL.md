---
name: ml-debugger-pro
description: Diagnosing and fixing ML training failures (NaNs, exploding gradients, dimension mismatches, data leakage). Use when the user reports a specific error or training artifact.
---

# ML-Debugger-Pro

This skill analyzes and fixes failures in deep learning models and pipelines.

## Workflow

1. **Failure Analysis:**
   - Use `grep_search` on the last `n` lines of the log file (`runs/.../history.json` or `wandb` logs).
   - Identify the "culprit" (e.g., loss divergence, shape mismatch, memory overflow).

2. **Diving Into Code:**
   - Use `read_file` on the reported `train.py` or `engine/...` script.
   - Look for common culprits: missing `zero_grad()`, lack of normalization, large learning rate, or unmasked `inf`/`nan`.

3. **Hypothesis & Fix:**
   - **Scenario 1 (NaNs):** Suggest `torch.autograd.set_detect_anomaly(True)` or gradient clipping.
   - **Scenario 2 (Slow convergence):** Recommend inspecting the learning rate scheduler or the data pipeline for bottleneck.
   - **Scenario 3 (Shape errors):** Use `read_file` on `forward()` to trace tensor shapes.

4. **Verification Strategy:**
   - Create a minimal "repro" script to confirm the fix before updating the full trainer.

## Direct Action Rules

- **Zero-Guessing:** If a user says "my loss is NaN," I must ask for the log or the specific architecture code immediately.
- **Trace-First:** If a shape error occurs, I must provide a `print(x.shape)` trace plan or a `pdb.set_trace()` instruction.
- **Hardware Aware:** If a `CUDA Out of Memory` error occurs, I must recommend batch size reduction or gradient accumulation steps (`effective_batch_size`).
