"""
engine/test.py
--------------
Smoke test for the full VasoJEPA model assembly.

Verifies:
  1. Model instantiates and parameter count is reported
  2. Full forward pass completes without crashing
  3. All three losses are non-NaN and non-zero
  4. backward() succeeds
  5. Gradients reach the encoder (proving JEPA loss flows correctly)
  6. Gradients reach predictor, LDS, and CGLT parameters

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

    # ── 1. Instantiate ────────────────────────────────────
    section("Initializing Model")
    model = Model().to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    enc_params   = sum(p.numel() for p in model.encoder.parameters())
    pred_params  = sum(p.numel() for p in model.predictor.parameters())
    lds_params   = sum(p.numel() for p in model.lds.parameters())
    cglt_params  = sum(p.numel() for p in model.cglt.parameters())

    print(f"  Total      : {total_params:>12,}")
    print(f"  Encoder    : {enc_params:>12,}")
    print(f"  Predictor  : {pred_params:>12,}")
    print(f"  LDS        : {lds_params:>12,}")
    print(f"  CGLT       : {cglt_params:>12,}")
    print(f"  Device     : {DEVICE}")

    # ── 2. Fake batch ─────────────────────────────────────
    section("Forward Pass")
    x     = torch.randn(BATCH, 1, 224, 224).to(DEVICE)
    prior = torch.rand(BATCH, 14, 14).to(DEVICE)

    model.train()
    loss, loss_dict = model(x, prior, epoch=0, total_epochs=100)

    print(f"  loss_dense : {loss_dict['dense']:.6f}")
    print(f"  loss_cglt  : {loss_dict['cglt']:.6f}")
    print(f"  loss_lds   : {loss_dict['lds']:.6f}")
    print(f"  loss_total : {loss.item():.6f}")

    # ── 3. Sanity checks ──────────────────────────────────
    section("Loss Sanity Checks")
    assert not torch.isnan(loss),     "Total loss is NaN!"
    assert loss.item() > 0,           "Total loss is zero — collapse?"
    for k, v in loss_dict.items():
        assert v == v,                f"{k} loss is NaN!"
        assert v > 0,                 f"{k} loss is zero!"
    print("  All losses non-NaN and non-zero [OK]")

    # ── 4. Backward ───────────────────────────────────────
    section("Gradient Flow Check")
    loss.backward()

    enc_grad  = next(model.encoder.parameters()).grad
    pred_grad = next(model.predictor.parameters()).grad
    lds_grad  = next(model.lds.parameters()).grad
    cglt_grad = next(model.cglt.parameters()).grad

    assert enc_grad  is not None, "Encoder received no gradients!"
    assert pred_grad is not None, "Predictor received no gradients!"
    assert lds_grad  is not None, "LDS received no gradients!"
    assert cglt_grad is not None, "CGLT received no gradients!"

    print(f"  Encoder  grad norm : {enc_grad.norm().item():.6f}  [OK]")
    print(f"  Predictor grad norm: {pred_grad.norm().item():.6f}  [OK]")
    print(f"  LDS      grad norm : {lds_grad.norm().item():.6f}  [OK]")
    print(f"  CGLT     grad norm : {cglt_grad.norm().item():.6f}  [OK]")

    section("ALL TESTS PASSED")
    print("  Full model assembly is working correctly.\n")


if __name__ == "__main__":
    run_test()
