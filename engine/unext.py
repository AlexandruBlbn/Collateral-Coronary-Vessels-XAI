"""UNeXt model factory used by training and inference scripts."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Sequence

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


def _load_unext_factory():
	"""Load UNeXt_S from zoo with a safe fallback for optional torchinfo."""
	try:
		from zoo.unext import UNeXt_S

		return UNeXt_S
	except ModuleNotFoundError as exc:
		if exc.name != "torchinfo":
			raise

		torchinfo_stub = types.ModuleType("torchinfo")
		torchinfo_stub.summary = lambda *args, **kwargs: None
		sys.modules.setdefault("torchinfo", torchinfo_stub)

		from zoo.unext import UNeXt_S

		return UNeXt_S


def get_model(
	in_channels: int = 1,
	num_classes: int = 1,
	base_channels: int = 32,
	depths: Sequence[int] = (2, 2, 1),
	mlp_ratio: int = 3,
	drop_rate: float = 0.0,
	attention: bool = True,
	use_checkpoint: bool = False,
	device: str | torch.device | None = None,
) -> nn.Module:
	"""Build and return a UNeXt-S model instance."""
	unext_factory = _load_unext_factory()
	model = unext_factory(
		in_channels=in_channels,
		num_classes=num_classes,
		base_channels=base_channels,
		depths=list(depths),
		mlp_ratio=mlp_ratio,
		drop_rate=drop_rate,
		attention=attention,
		use_checkpoint=use_checkpoint,
	)

	if device is not None:
		model = model.to(device)

	return model


if __name__ == "__main__":
	model = get_model()
	total_params = sum(param.numel() for param in model.parameters())
	print(f"Created model: {model.__class__.__name__}")
	print(f"Total parameters: {total_params:,}")
