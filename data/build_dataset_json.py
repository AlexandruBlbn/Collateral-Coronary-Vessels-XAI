#!/usr/bin/env python3
"""
Build a clean dataset.json from ARCADE processed data with train/val/test splits.

Split convention (per source):
  - First 1000   → train
  - Next 200     → validation  (1001-1200)
  - Next 300     → test        (1201-1500)
"""

import json
from pathlib import Path


def build_dataset_json(
    processed_dir: str = "data/ARCADE/processed",
    output_path: str = "data/ARCADE/processed/dataset.json",
    train_count: int = 1000,
    val_count: int = 200,
):
    processed_dir = Path(processed_dir)
    test_count = None  # computed as remainder

    # Discover source subdirectories (e.g. stenoza, syntax)
    sources = sorted([
        d.name for d in processed_dir.iterdir()
        if d.is_dir() and (d / "data").is_dir() and (d / "label").is_dir()
    ])

    print(f"Found sources: {sources}")

    dataset = {
        "train": {},
        "validation": {},
        "test": {},
    }

    for source in sources:
        data_dir = processed_dir / source / "data"
        label_dir = processed_dir / source / "label"

        # Get all PNG files sorted numerically by stem
        data_files = sorted(
            data_dir.glob("*.png"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
        )

        all_entries = []
        for df in data_files:
            stem = df.stem
            lf = label_dir / f"{stem}.png"
            if not lf.exists():
                print(f"  WARNING: {df.name} has no matching label, skipping")
                continue

            rel_data = str(Path("data/ARCADE/processed") / source / "data" / df.name).replace("\\", "/")
            rel_label = str(Path("data/ARCADE/processed") / source / "label" / lf.name).replace("\\", "/")

            all_entries.append({
                "data": rel_data,
                "label": rel_label,
            })

        # --- Split ---
        train_entries = {str(i + 1): e for i, e in enumerate(all_entries[:train_count])}
        val_entries = {
            str(i + 1 + train_count): e
            for i, e in enumerate(all_entries[train_count:train_count + val_count])
        }
        test_start = train_count + val_count
        test_entries = {
            str(i + 1 + test_start): e
            for i, e in enumerate(all_entries[test_start:])
        }

        dataset["train"][source] = train_entries
        dataset["validation"][source] = val_entries
        dataset["test"][source] = test_entries

        print(f"  {source}: {len(train_entries)} train / {len(val_entries)} val / {len(test_entries)} test")

    # Write
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nWritten to {output_path}")
    total = sum(
        sum(len(v) for v in split.values())
        for split in dataset.values()
    )
    print(f"Total samples: {total}")


if __name__ == "__main__":
    build_dataset_json()
