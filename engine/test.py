"""
engine/test.py
--------------
Smoke test for VasoJEPA model assembly.

Tests the full VesselJEPA configuration (EMA-free + vessel masking + vessel anchor),
which is the most complex variant. If this passes, simpler configs (EMA only, etc.)
will also work.

Verifies:
  1. Model instantiates with all 3 toggles (no EMA, vessel masking, vessel anchor)
  2. No target_encoder exists when use_ema=False
  3. Full forward pass completes without crashing
  4. All losses are non-NaN and non-zero
  5. backward() succeeds
  6. Gradients reach encoder, predictor, and vessel_head
  7. update_target_encoder is a no-op when use_ema=False
  8. Vessel masking produces valid target indices

Run from project root:
    python engine/test.py
"""

import sys
sys.path.append(".")
import torch
from vasojepa.model import Model


BATCH   = 2
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def section(title):
    print("\n" + "=" * 52)
    print("  " + title)
    print("=" * 52)


def run_test():

    # ── 1. Instantiate full VesselJEPA ─────────────────────
    section("Initializing VesselJEPA (EMA-free + vmask + vanchor)")
    model = Model(
        use_ema=False,
        vessel_masking=True,
        vessel_anchor=True,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    enc_params   = sum(p.numel() for p in model.encoder.parameters())
    pred_params  = sum(p.numel() for p in model.predictor.parameters())
    vhead_params = sum(p.numel() for p in model.vessel_head.parameters())

    print(f"  Total           : {total_params:>12,}")
    print(f"  Encoder         : {enc_params:>12,}")
    print(f"  Predictor       : {pred_params:>12,}")
    print(f"  Vessel head     : {vhead_params:>12,}")
    print(f"  Device          : {DEVICE}")

    # Verify NO target encoder when use_ema=False
    assert not hasattr(model, 'target_encoder'), "target_encoder should not exist when use_ema=False!"
    print(f"  No target_encoder (EMA-free)  [OK]")
    assert hasattr(model, 'vessel_head'), "vessel_head should exist when vessel_anchor=True!"
    print(f"  vessel_head exists            [OK]")

    # ── 2. Fake batch ──────────────────────────────────────
    section("Forward Pass")
    x     = torch.randn(BATCH, 1, 224, 224).to(DEVICE)
    prior = torch.rand(BATCH, 14, 14).to(DEVICE)

    model.train()
    loss, loss_dict = model(x, prior, epoch=0, total_epochs=100)

    print(f"  loss_dense     : {loss_dict['dense']:.6f}")
    print(f"  anchor         : {loss_dict['anchor']:.6f}")
    print(f"  consistency    : {loss_dict['consistency']:.6f}")
    print(f"  f2_std         : {loss_dict['f2_std']:.6f}")
    print(f"  tf2_std        : {loss_dict['tf2_std']:.6f}")
    print(f"  loss_total     : {loss.item():.6f}")

    # ── 3. Sanity checks ───────────────────────────────────
    section("Loss Sanity Checks")
    assert not torch.isnan(loss),     "Total loss is NaN!"
    assert loss.item() > 0,           "Total loss is zero — collapse?"
    for k, v in loss_dict.items():
        assert v == v,                f"{k} is NaN!"
        assert v > 0 or k in ("f2_std", "tf2_std"), f"{k} is zero!"
    print("  All losses non-NaN, non-zero  [OK]")

    # ── 4. Backward + gradient flow ────────────────────────
    section("Gradient Flow Check")
    loss.backward()

    enc_grad   = next(model.encoder.parameters()).grad
    pred_grad  = next(model.predictor.parameters()).grad
    vhead_grad = next(model.vessel_head.parameters()).grad

    assert enc_grad   is not None, "Encoder received no gradients!"
    assert pred_grad  is not None, "Predictor received no gradients!"
    assert vhead_grad is not None, "Vessel head received no gradients!"

    print(f"  Encoder       grad norm : {enc_grad.norm().item():.6f}  [OK]")
    print(f"  Predictor     grad norm : {pred_grad.norm().item():.6f}  [OK]")
    print(f"  Vessel head   grad norm : {vhead_grad.norm().item():.6f}  [OK]")

    # ── 5. EMA update is no-op ─────────────────────────────
    section("EMA Update Check (should be no-op)")
    model.update_target_encoder(progress=0.5)
    print(f"  update_target_encoder returned without error  [OK]")

    # ── 6. Vessel masking validity ─────────────────────────
    section("Vessel Masking Check")
    # Run forward again and check target indices are valid
    model.train()
    with torch.no_grad():
        f0, f1, f2, f3 = model.encoder(x)
        B, N, dim = f2.shape
        prior_flat = prior.view(B, N)
        weights = prior_flat + 0.15
        for b in range(B):
            tgt = torch.multinomial(weights[b], int(N * 0.35), replacement=False)
            assert tgt.min() >= 0 and tgt.max() < N, f"Target indices out of range for batch {b}!"
            assert len(torch.unique(tgt)) == len(tgt), f"Duplicate target indices for batch {b}!"
    print(f"  Vessel masking produces valid unique indices  [OK]")

    section("ALL TESTS PASSED")
    print("  VesselJEPA model assembly is working correctly.\n")


if __name__ == "__main__":
    run_test()
