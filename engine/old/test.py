import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from engine.train_targeted_vessel_segmentation import TargetedSyntaxSegmentationDataset


def save_four_channel_plot(
    target_csv="results/arcade_patient_tables/patient_main_artery_targets.csv",
    syntax_root="data/ARCADE/Unprocessed/arcade/syntax",
    split="train",
    img_size=512,
    sample_idx=0,
    out_path="channel_plot.png",
):
    ds = TargetedSyntaxSegmentationDataset(
        target_csv=target_csv,
        syntax_root=syntax_root,
        split=split,
        img_size=img_size,
        mode="train" if split == "train" else "val",
    )

    img_t, mask_t, aux_t, centerline_t, vector_t, target_id, file_name = ds[sample_idx]

    channel_names = ["CLAHE", "White-hat / Top-hat", "Black-hat", "Sharpness / High-pass"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f"{file_name} | target_id={target_id}", fontsize=12)

    for i, ax in enumerate(axes):
        channel = img_t[i].detach().cpu().numpy()
        ax.imshow(channel, cmap="gray")
        ax.set_title(channel_names[i])
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    save_four_channel_plot()