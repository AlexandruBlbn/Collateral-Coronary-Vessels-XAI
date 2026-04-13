"""
CoronaryDominance Dataset — Pretraining Loader
================================================
160,320 grayscale X-ray angiography frames (512×512) extracted from
1,574 patients. Source: https://huggingface.co/datasets/BearSubj13/CoronaryDominance

Naming pattern:
    StudyXXXXX_[LCA|RCA]_sequenceY_sliceZ.png

Two classes are provided:

1.  CoronaryDominanceDataset     — flat pretrain dataset (no labels),
                                   compatible with DINOv3Augmentation.
2.  SequenceCoronaryDominance    — yields temporal windows of T consecutive
                                   frames from the same angiography sequence,
                                   useful for temporal contrastive pretraining.

Quick usage with DINOv3Augmentation::

    from data.CoronaryDominance import CoronaryDominanceDataset
    from data.DinoV3 import DINOv3Augmentation

    augmenter = DINOv3Augmentation()
    ds = CoronaryDominanceDataset(
        root_dir="XA-170K/dataset/coronarydominance",
        transform=augmenter,
    )
    loader = DataLoader(ds, batch_size=32, num_workers=8, collate_fn=dino_collate)

Quick usage with JSON manifest (patient split)::

    from data.CoronaryDominance import CoronaryDominanceDataset
    ds = CoronaryDominanceDataset.from_json(
        json_path="data/ARCADE/processed/dataset.json",
        split="train",
        root_dir=".",
    )
"""

import os
import re
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Optional, Callable, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Naming pattern helpers
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(
    r"(?P<study>Study\d+)_(?P<vessel>LCA|RCA)_sequence(?P<seq>\d+)_slice(?P<slc>\d+)\.png"
)


def _parse_filename(fname: str) -> Optional[dict]:
    """Return parsed fields or None if the filename does not match."""
    m = _FNAME_RE.match(fname)
    if m is None:
        return None
    return {
        "study":    m.group("study"),
        "vessel":   m.group("vessel"),
        "sequence": int(m.group("seq")),
        "slice":    int(m.group("slc")),
        "fname":    fname,
    }


# ---------------------------------------------------------------------------
# 1. Flat pretrain dataset
# ---------------------------------------------------------------------------

class CoronaryDominanceDataset(Dataset):
    """
    Flat, label-free dataset over all CoronaryDominance frames.

    Args:
        root_dir (str | Path):
            Directory containing the PNG files
            (``XA-170K/dataset/coronarydominance``).
        file_list (list[str] | None):
            Optional explicit list of *filenames* (not full paths) to use.
            If None, all ``*.png`` files in ``root_dir`` are used.
        transform (callable | None):
            Transform applied to each PIL Image.
            Pass a ``DINOv3Augmentation()`` instance for DINO pretraining.
        return_meta (bool):
            If True, ``__getitem__`` returns ``(image, meta_dict)`` where
            ``meta_dict`` contains study / vessel / sequence / slice info.
            Useful for debugging; leave False for training.
    """

    def __init__(
        self,
        root_dir: str | Path,
        file_list: Optional[List[str]] = None,
        transform:  Optional[Callable] = None,
        return_meta: bool = False,
    ):
        self.root_dir    = Path(root_dir)
        self.transform   = transform
        self.return_meta = return_meta

        if file_list is not None:
            self.samples = list(file_list)
        else:
            self.samples = sorted(
                f for f in os.listdir(self.root_dir) if f.endswith(".png")
            )

        if not self.samples:
            raise RuntimeError(
                f"No PNG files found in {self.root_dir}.  "
                "Check that the path is correct."
            )

    # ------------------------------------------------------------------
    # Alternative constructor — build from existing dataset.json
    # ------------------------------------------------------------------

    @classmethod
    def from_json(
        cls,
        json_path: str | Path,
        split: str = "train",
        root_dir: str | Path = ".",
        transform: Optional[Callable] = None,
        return_meta: bool = False,
    ) -> "CoronaryDominanceDataset":
        """
        Build a CoronaryDominanceDataset from the project's dataset.json.

        The JSON must contain::

            {split: {"coronarydominance": {"id": {"data": "relative/path.png"}, ...}}}

        Args:
            json_path: Path to ``dataset.json``.
            split: One of ``"train"``, ``"validation"``, ``"test"``.
            root_dir: Workspace root used to resolve relative paths stored in
                      the JSON (typically the project root).
            transform: Optional transform.
        """
        json_path = Path(json_path)
        root_dir  = Path(root_dir)

        with open(json_path) as f:
            data = json.load(f)

        if split not in data:
            raise ValueError(f"Split '{split}' not found in {json_path}.")

        source = data[split].get("coronarydominance", {})
        if not source:
            raise ValueError(
                f"'coronarydominance' source not found in split '{split}' of {json_path}.\n"
                "Run the JSON-generation script first."
            )

        # Each value is {"data": "XA-170K/dataset/coronarydominance/StudyXXX...png", "label": ""}
        abs_paths = []
        for entry in source.values():
            rel = entry.get("data", "")
            if rel:
                abs_paths.append(str(root_dir / rel))

        # We use full absolute paths — root_dir of the Dataset itself is "/"
        instance = cls.__new__(cls)
        instance.root_dir    = Path("/")   # paths are already absolute
        instance.transform   = transform
        instance.return_meta = return_meta
        instance.samples     = abs_paths
        return instance

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # Handle both relative filenames and absolute paths
        if os.path.isabs(sample):
            img_path = Path(sample)
        else:
            img_path = self.root_dir / sample

        image = Image.open(img_path).convert("L")   # grayscale

        if self.transform is not None:
            image = self.transform(image)

        if self.return_meta:
            fname = img_path.name
            meta  = _parse_filename(fname) or {"fname": fname}
            return image, meta

        return image

    def __repr__(self) -> str:
        return (
            f"CoronaryDominanceDataset("
            f"n={len(self)}, root={self.root_dir}, "
            f"transform={self.transform.__class__.__name__ if self.transform else None})"
        )


# ---------------------------------------------------------------------------
# 2. Sequence-aware temporal dataset (optional, for temporal SSL)
# ---------------------------------------------------------------------------

class SequenceCoronaryDominance(Dataset):
    """
    Yield windows of **T consecutive frames** from the same angiography
    sequence (same study + vessel + sequence number).

    Each item is a list of T PIL Images (or tensors if a transform is given).

    This is useful for temporal contrastive objectives like:
        - BYOL / SimCLR with adjacent-frame positive pairs
        - VideoMAE-style masked pretraining over short temporal clips
        - Temporal consistency regularisation

    Args:
        root_dir: Directory with all PNG frames.
        T: Window length (number of consecutive frames).
        stride: Step between windows (1 = dense, T = non-overlapping).
        transform: Applied independently to each frame in the window.
        min_seq_len: Sequences shorter than this are skipped.
    """

    def __init__(
        self,
        root_dir: str | Path,
        T: int = 4,
        stride: int = 1,
        transform: Optional[Callable] = None,
        min_seq_len: int = 4,
    ):
        self.root_dir  = Path(root_dir)
        self.T         = T
        self.stride    = stride
        self.transform = transform

        # Group frames by (study, vessel, sequence)
        seq_groups: dict[tuple, list] = defaultdict(list)

        for fname in os.listdir(self.root_dir):
            meta = _parse_filename(fname)
            if meta is None:
                continue
            key = (meta["study"], meta["vessel"], meta["sequence"])
            seq_groups[key].append(meta)

        # Sort each group by slice index and build windows
        self.windows: List[List[str]] = []
        for key, frames in seq_groups.items():
            frames.sort(key=lambda m: m["slice"])
            slices = [f["fname"] for f in frames]
            if len(slices) < min_seq_len:
                continue
            for start in range(0, len(slices) - T + 1, stride):
                self.windows.append(slices[start : start + T])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> list:
        filenames = self.windows[idx]
        frames = []
        for fname in filenames:
            img = Image.open(self.root_dir / fname).convert("L")
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)
        return frames

    def __repr__(self) -> str:
        return (
            f"SequenceCoronaryDominance("
            f"windows={len(self)}, T={self.T}, stride={self.stride}, "
            f"root={self.root_dir})"
        )


# ---------------------------------------------------------------------------
# Utility: random patient-split file lists (if you prefer .txt over JSON)
# ---------------------------------------------------------------------------

def make_split_file_lists(
    root_dir:   str | Path,
    out_dir:    str | Path,
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    seed:        int   = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Scan ``root_dir``, split by patient (study_id), write three .txt files::

        out_dir/cd_train.txt
        out_dir/cd_val.txt
        out_dir/cd_test.txt

    Each line is a relative path:  ``XA-170K/dataset/coronarydominance/<fname>``

    Returns:
        (train_fnames, val_fnames, test_fnames)
    """
    root_dir = Path(root_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_study: dict[str, list] = defaultdict(list)
    for fname in os.listdir(root_dir):
        meta = _parse_filename(fname)
        if meta:
            by_study[meta["study"]].append(fname)

    studies = sorted(by_study.keys())
    rng = random.Random(seed)
    rng.shuffle(studies)

    n = len(studies)
    n_val   = max(1, int(n * val_ratio))
    n_test  = max(1, int(n * (1.0 - train_ratio - val_ratio)))
    n_train = n - n_val - n_test

    splits = {
        "train": studies[:n_train],
        "val":   studies[n_train : n_train + n_val],
        "test":  studies[n_train + n_val :],
    }

    results = {}
    prefix = f"XA-170K/dataset/coronarydominance"
    for split_name, study_list in splits.items():
        fnames = []
        for s in sorted(study_list):
            fnames.extend(sorted(by_study[s]))
        out_file = out_dir / f"cd_{split_name}.txt"
        out_file.write_text("\n".join(f"{prefix}/{f}" for f in fnames))
        results[split_name] = fnames
        print(f"Wrote {len(fnames):>7,} entries → {out_file}")

    return results["train"], results["val"], results["test"]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    root = Path(
        sys.argv[1] if len(sys.argv) > 1
        else "XA-170K/dataset/coronarydominance"
    )

    print("=== CoronaryDominanceDataset (flat) ===")
    ds = CoronaryDominanceDataset(root_dir=root)
    print(ds)
    img = ds[0]
    print(f"  Sample 0 type  : {type(img)}")
    print(f"  Sample 0 size  : {img.size}  mode={img.mode}")

    print("\n=== SequenceCoronaryDominance (T=4) ===")
    seq_ds = SequenceCoronaryDominance(root_dir=root, T=4, stride=2)
    print(seq_ds)
    window = seq_ds[0]
    print(f"  Window[0] len  : {len(window)}")
    print(f"  Frame[0] size  : {window[0].size}")
