import argparse
import copy
import csv
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

import trainv4 as t4


def build_base_config() -> Dict:
	return {
		"experiment_name": "trainv4_effnetv2s_spcl",
		"logging": {
			"log_dir": "runs/{experiment_name}",
			"checkpoint_dir": "checkpoints/{experiment_name}",
			"visualize_every_epochs": 2,
		},
		"data": {
			"json_path": "data/ARCADE/processed/dataset.json",
			"root_dir": ".",
			"source": "syntax",
			"img_size": 512,
			"batch_size": 4,
			"num_workers": 12,
			"pin_memory": True,
			"prefetch_factor": 4,
			"sample_weights_csv": "results/hard_case_mining/sample_weights_train.csv",
		},
		"training": {
			"epochs": 200,
			"learning_rate": 2e-4,
			"weight_decay": 1e-4,
			"scheduler": "Warmup + CosineAnnealingLR",
			"precision": "bfloat16",
			"use_amp": True,
			"accum_steps": 2,
			"clip_grad_norm": 1.0,
			"patience": 50,
			"warmup_epochs": 5,
		},
		"model": {
			"name": "VesselNetV3EfficientNet",
			"in_chans": 4,
			"num_classes": 1,
			"encoder_name": "efficientnetv2_s",
			"encoder_pretrained": False,
			"encoder_img_size": 512,
			"embedding_dim": 128,
			"drop_path_rate": 0.3,
		},
		"loss": {
			"recipe": "OHEM BCE + Faster Boundary Annealing + clDice + FocalTversky + SPCL + Deep Supervision + SDM",
			"ohem_ratio": 0.25,
			"ohem_min_kept": 2048,
			"cldice_weight": 0.2,
			"focal_tversky_weight": 0.25,
			"spcl_weight": 0.2,
			"sce_weight": 1.0,
			"pcl_weight": 1.0,
			"spcl_temperature": 0.1,
			"spcl_num_prototypes": 2,
			"spcl_margin": 0.7,
			"spcl_pos_weight": 1.0,
			"spcl_neg_weight": 1.0,
			"spcl_hard_neg_ratio": 0.25,
			"spcl_max_samples": 1024,
			"tversky_alpha": 0.3,
			"tversky_beta": 0.7,
			"tversky_gamma": 2.0,
			"deep_supervision_weight": 0.2,
			"sdm_weight": 0.25,
			"bce_start_weight": 1.0,
			"bce_end_weight": 0.2,
			"boundary_start_weight": 0.0,
			"boundary_end_weight": 1.0,
			"anneal_epochs": 30,
			"anneal_power": 0.7,
			"cldice_iters": 5,
		},
		"evaluation": {
			"default_threshold": 0.5,
			"threshold_grid": [round(float(x), 2) for x in np.arange(0.1, 0.95, 0.05)],
		},
	}


def build_weighted_sampler_seeded(
	dataset: t4.VesselSegmentationDatasetV2,
	sample_weights_csv: str,
	seed: int,
) -> WeightedRandomSampler:
	if not sample_weights_csv or not os.path.isfile(sample_weights_csv):
		return None

	weight_dict = {}
	with open(sample_weights_csv, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			file_name = row.get("file_name", "")
			value = float(row.get("sample_weight", 1.0))
			weight_dict[file_name] = max(1e-3, value)

	sample_weights = [weight_dict.get(sample["file_name"], 1.0) for sample in dataset.samples]
	weights_tensor = torch.tensor(sample_weights, dtype=torch.double)

	gen = torch.Generator()
	gen.manual_seed(int(seed))
	try:
		return WeightedRandomSampler(
			weights=weights_tensor,
			num_samples=len(sample_weights),
			replacement=True,
			generator=gen,
		)
	except TypeError:
		# Compatibility fallback for older torch versions without generator support.
		return WeightedRandomSampler(weights=weights_tensor, num_samples=len(sample_weights), replacement=True)


def seeded_loader(
	img_size: int,
	batch_size: int,
	split: str,
	config: Dict,
	seed: int,
	sampler: WeightedRandomSampler = None,
):
	def seed_worker(worker_id):
		worker_seed = torch.initial_seed() % 2**32
		np.random.seed(worker_seed)
		random.seed(worker_seed)

	ds = t4.VesselSegmentationDatasetV2(
		json_path=config["data"]["json_path"],
		split=split,
		source=config["data"].get("source", "syntax"),
		img_size=img_size,
		mode="train" if split == "train" else "eval",
		root_dir=config["data"].get("root_dir", "."),
	)

	g = torch.Generator()
	split_offset = {"train": 0, "validation": 101, "val": 101, "test": 202}.get(split, 303)
	g.manual_seed(int(seed) + int(split_offset))

	num_workers = int(config["data"].get("num_workers", 4))
	pin_memory = bool(config["data"].get("pin_memory", torch.cuda.is_available()))
	prefetch_factor = int(config["data"].get("prefetch_factor", 4))

	loader_kwargs = {
		"batch_size": batch_size,
		"shuffle": (split == "train" and sampler is None),
		"sampler": sampler,
		"num_workers": num_workers,
		"persistent_workers": (num_workers > 0),
		"pin_memory": pin_memory,
		"worker_init_fn": seed_worker,
		"generator": g,
	}
	if num_workers > 0:
		loader_kwargs["prefetch_factor"] = max(2, prefetch_factor)

	return DataLoader(ds, **loader_kwargs)


def build_model(config: Dict):
	model_name = str(config["model"].get("name", "VesselNetV2"))
	if model_name == "VesselNetV2":
		model = t4.VesselNetV2(
			in_chans=int(config["model"]["in_chans"]),
			num_classes=int(config["model"]["num_classes"]),
			dims=tuple(config["model"].get("dims", [48, 96, 192, 384])),
			depths=tuple(config["model"].get("depths", [2, 2, 2, 2])),
			drop_path_rate=float(config["model"].get("drop_path_rate", 0.1)),
		).to(t4.device)
	elif model_name == "VesselNetV3EfficientNet":
		model = t4.VesselNetV3EfficientNet(
			in_chans=int(config["model"]["in_chans"]),
			num_classes=int(config["model"]["num_classes"]),
			encoder_name=str(config["model"].get("encoder_name", "efficientnetv2_s")),
			encoder_pretrained=bool(config["model"].get("encoder_pretrained", False)),
			encoder_img_size=int(config["model"].get("encoder_img_size", config["data"].get("img_size", 512))),
			embedding_dim=int(config["model"].get("embedding_dim", 128)),
			drop_path_rate=float(config["model"].get("drop_path_rate", 0.1)),
		).to(t4.device)
	else:
		raise ValueError(f"Unknown model name: {model_name}")

	return model


def train_one_seed(config_template: Dict, seed: int, experiment_name: str, epochs_override: int = 0) -> Tuple[str, float]:
	config = copy.deepcopy(config_template)
	config["experiment_name"] = experiment_name
	if int(epochs_override) > 0:
		config["training"]["epochs"] = int(epochs_override)

	t4.set_seed(int(seed))
	config["training"]["seed"] = int(seed)

	writer = SummaryWriter(log_dir=config["logging"]["log_dir"].format(experiment_name=config["experiment_name"]))

	train_dataset = t4.VesselSegmentationDatasetV2(
		json_path=config["data"]["json_path"],
		split="train",
		source=config["data"].get("source", "syntax"),
		img_size=config["data"]["img_size"],
		mode="train",
		root_dir=config["data"].get("root_dir", "."),
	)
	sampler = build_weighted_sampler_seeded(train_dataset, config["data"].get("sample_weights_csv", ""), seed=int(seed))

	train_loader = seeded_loader(
		img_size=config["data"]["img_size"],
		batch_size=config["data"]["batch_size"],
		split="train",
		config=config,
		seed=int(seed),
		sampler=sampler,
	)
	val_loader = seeded_loader(
		img_size=config["data"]["img_size"],
		batch_size=config["data"]["batch_size"],
		split="validation",
		config=config,
		seed=int(seed),
		sampler=None,
	)
	test_loader = seeded_loader(
		img_size=config["data"]["img_size"],
		batch_size=config["data"]["batch_size"],
		split="test",
		config=config,
		seed=int(seed),
		sampler=None,
	)

	model = build_model(config)

	optimiser = optim.AdamW(
		model.parameters(),
		lr=float(config["training"]["learning_rate"]),
		weight_decay=float(config["training"].get("weight_decay", 1e-4)),
	)

	warmup_epochs = int(config["training"].get("warmup_epochs", 5))
	total_epochs = int(config["training"]["epochs"])
	warmup_epochs = min(warmup_epochs, max(1, total_epochs - 1))

	warmup = LinearLR(optimiser, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
	cosine = CosineAnnealingLR(optimiser, T_max=max(1, total_epochs - warmup_epochs))
	scheduler = SequentialLR(optimiser, schedulers=[warmup, cosine], milestones=[warmup_epochs])

	criterion = t4.VesselHybridLoss(config=config)

	t4.configCreate(
		os.path.join(
			config["logging"]["log_dir"].format(experiment_name=config["experiment_name"]),
			"config.yaml",
		),
		config,
	)

	try:
		best_model_path, best_test_f1 = t4.trainScript(
			model=model,
			train_loader=train_loader,
			val_loader=val_loader,
			test_loader=test_loader,
			criterion=criterion,
			optimiser=optimiser,
			scheduler=scheduler,
			num_epochs=total_epochs,
			config=config,
			tb_writer=writer,
		)
	finally:
		writer.close()

	return best_model_path, float(best_test_f1)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run trainv4 across multiple seeds and summarize results.")
	parser.add_argument(
		"--seeds",
		nargs="+",
		type=int,
		default=[42, 1337, 2025],
		help="List of seeds to train.",
	)
	parser.add_argument(
		"--experiment-prefix",
		type=str,
		default="trainv4_effnetv2s_spcl_multiseed",
		help="Prefix used to create per-seed experiment_name.",
	)
	parser.add_argument(
		"--tag",
		type=str,
		default="",
		help="Optional run tag appended to experiment names and summary CSV.",
	)
	parser.add_argument(
		"--allow-resume",
		action="store_true",
		help="Allow reusing existing per-seed experiment names, which may resume from checkpoints.",
	)
	parser.add_argument(
		"--epochs",
		type=int,
		default=0,
		help="Override training epochs if > 0.",
	)
	parser.add_argument(
		"--results-dir",
		type=str,
		default="results/seed_sweeps",
		help="Directory for aggregate multi-seed summary CSV.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	base_config = build_base_config()

	tag = str(args.tag).strip()
	if (not args.allow_resume) and (not tag):
		tag = datetime.now().strftime("%Y%m%d_%H%M%S")
		print(f"[INFO] Auto tag '{tag}' enabled to avoid accidental checkpoint resume.")

	rows: List[Dict[str, str]] = []
	for seed in args.seeds:
		experiment_name = f"{args.experiment_prefix}_seed{int(seed)}"
		if tag:
			experiment_name = f"{experiment_name}_{tag}"

		print("=" * 80)
		print(f"[INFO] Starting seed={int(seed)} | experiment={experiment_name}")
		print("=" * 80)

		best_model_path, best_test_f1 = train_one_seed(
			config_template=base_config,
			seed=int(seed),
			experiment_name=experiment_name,
			epochs_override=int(args.epochs),
		)

		rows.append(
			{
				"seed": str(int(seed)),
				"experiment_name": experiment_name,
				"best_model_path": best_model_path,
				"test_f1_youden": f"{best_test_f1:.6f}",
			}
		)

		print(f"[INFO] Finished seed={int(seed)} | test_f1_youden={best_test_f1:.6f}")

	os.makedirs(args.results_dir, exist_ok=True)
	summary_name = args.experiment_prefix
	if tag:
		summary_name = f"{summary_name}_{tag}"
	summary_path = os.path.join(args.results_dir, f"{summary_name}.csv")

	with open(summary_path, "w", newline="", encoding="utf-8") as f:
		fieldnames = ["seed", "experiment_name", "best_model_path", "test_f1_youden"]
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)

	f1_values = np.array([float(row["test_f1_youden"]) for row in rows], dtype=np.float32)
	mean_f1 = float(np.mean(f1_values)) if f1_values.size > 0 else 0.0
	std_f1 = float(np.std(f1_values)) if f1_values.size > 1 else 0.0

	print("\n" + "=" * 80)
	print("Multi-seed summary")
	print("=" * 80)
	print(f"Seeds: {[int(s) for s in args.seeds]}")
	print(f"Mean test F1 (Youden): {mean_f1:.6f}")
	print(f"Std  test F1 (Youden): {std_f1:.6f}")
	print(f"Saved: {summary_path}")


if __name__ == "__main__":
	main()
