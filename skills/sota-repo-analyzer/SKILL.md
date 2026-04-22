---
name: sota-repo-analyzer
description: Deep-diving into ML repositories to extract implementation tricks and non-obvious details. Use when the user wants to understand a specific repository or "the way SOTA works" for a task.
---

# SOTA-Repo-Analyzer

This skill maps and explains the internals of state-of-the-art repositories (GitHub, GitLab, HF).

## Workflow

1. **Repo Discovery:**
   - Search `google_web_search` for the repo URL if not provided.
   - Use `web_fetch` on the `README.md` to find the "Main" script/entry point.

2. **Logic Mapping:**
   - Use `web_fetch` on raw files to find the core `class Model(nn.Module):` and `def forward(self, ...):`.
   - Identify "Tricks" (non-standard normalization, custom `torch.autograd.Function`, specific weight init).

3. **Comparison:**
   - Compare the repo implementation against the paper's claims.
   - Flag discrepancies (e.g., "The paper says LR=1e-4, but the code says 5e-5").

4. **Integration Guide:**
   - Summarize how to import this model into the current project's `zoo/` folder.

## Direct Action Rules

- **Code-First:** If a user asks "How does it work?", provide the exact code block from the repo.
- **Dependency Map:** List all non-standard dependencies (`timm`, `einops`, `cupy`, `triton`) found in the repo's `requirements.txt`.
- **Repo-to-Project:** If the user likes the implementation, propose a direct `write_file` command to port the logic to `zoo/<new_model>.py`.
