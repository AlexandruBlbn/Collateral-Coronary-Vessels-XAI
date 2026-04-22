---
name: paper-to-prototype
description: Researching ML/DL papers, extracting math/architecture, and creating "Research Memories" for prototype implementation. Use when the user mentions an ArXiv ID, paper title, or a specific model architecture to research.
---

# Paper-to-Prototype

This skill transforms research papers (ArXiv, CVPR, NeurIPS) into structured "Research Memories" and implementation-ready prototypes.

## Workflow

1. **Research & Extract:**
   - Use `google_web_search` to find the PDF and the official/community GitHub repository.
   - Use `web_fetch` on the paper PDF/HTML to extract:
     - Mathematical definitions (LaTeX).
     - Architecture details (Layers, initialization, normalization).
     - Hyperparameters (Optimizer, learning rate schedules, augmentations).
     - Training tricks (Mixed precision, gradient clipping, EMA).

2. **Create Research Memory:**
   - Create a file at `docs/research_memories/<paper_name>.md`.
   - **Structure:**
     - **Abstract & Impact:** Why this paper matters.
     - **Key Contributions:** The "secret sauce" (e.g., a new loss function or attention mechanism).
     - **Math & Logic:** Detailed LaTeX equations and pseudocode.
     - **Implementation Specs:** Input/output shapes, parameter counts, and config values.
     - **Repo Links:** Official and highly-starred community implementations.

3. **Generate Prototype:**
   - Generate a `zoo/prototypes/<model_name>.py` file with the PyTorch/JAX implementation.
   - Ensure it follows the project's existing coding standards (e.g., matching the style in `zoo/backbones.py`).

## Direct Action Rules

- **Zero-Guessing:** If an equation or hyperparameter is missing from the abstract, search the full text or the official repo immediately.
- **Repository-First:** If a repository exists, prioritize its `forward()` implementation over the paper's simplified text description.
- **Dependency Check:** Before proposing an implementation, check `requirements.txt` to see if libraries like `timm` or `einops` are already available.
