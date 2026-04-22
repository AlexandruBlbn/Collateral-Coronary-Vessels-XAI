---
name: ml-experiment-architect
description: Scaffolding DL experiments (Trainer scripts, configs, datasets) based on the current workspace architecture. Use when the user wants to start a new experiment or build a baseline.
---

# ML-Experiment-Architect

This skill generates and maintains the "scaffolding" (the non-research code) for ML/DL projects.

## Workflow

1. **Context Mapping:**
   - Use `list_directory` on `config/`, `engine/`, and `data/` to identify established patterns.
   - Example: If most trainers use PyTorch Lightning or `wandb`, this skill must use them too.

2. **Dataloader Generation:**
   - Analyze `data/` scripts (like `data/dataloader.py`) to create compatible wrappers.
   - Always include standard augmentations (Albumentations/Torchvision) as part of the boilerplate.

3. **YAML Config Design:**
   - Create or update a `config/<experiment_name>.yaml` file.
   - **Fields:** Architecture name, dataset paths, training args, and log settings.

4. **Direct Shell Execution:**
   - Propose a command to run the experiment (e.g., `python engine/train.py --config config/exp_001.yaml`).

## Direct Action Rules

- **Zero Boilerplate:** Never generate a "hello world" trainer. Always use the project's real data classes and backbone libraries.
- **Dependency Guard:** If a user asks for a feature like `mixed_precision` but `torch` version is old, check `pip list` or `requirements.txt` first.
- **Log-Ready:** Always include at least one logging mechanism (WandB, Tensorboard, or CSV) in the generated trainer.
