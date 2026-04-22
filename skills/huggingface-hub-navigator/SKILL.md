---
name: huggingface-hub-navigator
description: Automating model and dataset discovery from HuggingFace. Use when the user asks for a specific HF model or a dataset to integrate into their pipeline.
---

# HuggingFace-Hub-Navigator

This skill manages the discovery and integration of models, datasets, and spaces from the HuggingFace Hub.

## Workflow

1. **Discovery & Search:**
   - Search `google_web_search` for `site:huggingface.co <model/dataset name>` to find the direct URL.
   - Use `web_fetch` on the model/dataset card to find:
     - The exact ID (e.g., `google/vit-base-patch16-224`).
     - Supported frameworks (PyTorch, TensorFlow, JAX).
     - Usage snippets (e.g., `AutoModel`, `datasets.load_dataset`).

2. **Setup & Integration:**
   - Propose a command to install required libraries (`pip install transformers datasets hub`).
   - Generate a boilerplate script in `data/hf_datasets/<dataset_name>.py` or `zoo/hf_models/<model_name>.py`.
   - Map HF parameters (e.g., `num_labels`, `id2label`) to the current project's config.

3. **Inference Preview:**
   - If a `Space` or `Inference Widget` is available, describe the input/output format and provide a `curl` example for the HF Inference API.

## Direct Action Rules

- **Zero-Boilerplate:** Don't just give the snippet; adapt it to the project's existing dataloader/backbone structure.
- **Cache Management:** If downloading a large model, warn the user and check available disk space or environment variables (`HF_HOME`).
- **Framework Check:** Only suggest HF models that match the project's primary framework (likely PyTorch, based on `zoo/` structure).
