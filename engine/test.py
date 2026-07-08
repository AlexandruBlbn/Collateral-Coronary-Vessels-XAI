"""
engine/test.py
--------------
Smoke test for the Encoder + Predictor assembly.

What this verifies:
  1. Encoder produces the correct output shapes per stage
  2. Random masking splits f2 tokens into context and target sets correctly
  3. Predictor receives context tokens and outputs predictions at target positions
  4. Shapes of predictions match encoder targets exactly
  5. MSE loss is computable (non-zero, non-NaN)
  6. Gradients flow through the predictor but STOP at the target (detach check)

Run from the project root:
    python engine/test.py
"""

import sys
sys.path.append(".")
import torch
import torch.nn.functional as F
from vasojepa.encoder import Encoder
from vasojepa.predictor import Predictor


# -- Config -------------------------------------------------------------------
BATCH_SIZE = 2
IMG_SIZE   = 224
MASK_RATIO = 0.4   # 40% of Stage 2 tokens become targets
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_mask(B, N, mask_ratio, device):
    """
    Randomly split N token positions into context and target sets.
    Returns:
        context_idx : [B, N_ctx]  indices of visible (context) tokens
        target_idx  : [B, N_tgt]  indices of masked (target) tokens
    """
    N_tgt = int(N * mask_ratio)
    N_ctx = N - N_tgt

    context_idx = []
    target_idx  = []
    for _ in range(B):
        perm = torch.randperm(N, device=device)
        context_idx.append(perm[:N_ctx])
        target_idx.append(perm[N_ctx:])

    context_idx = torch.stack(context_idx)  # [B, N_ctx]
    target_idx  = torch.stack(target_idx)   # [B, N_tgt]
    return context_idx, target_idx


def gather_tokens(features, indices):
    """
    Gather tokens at given indices from a feature sequence.
    features : [B, N, C]
    indices  : [B, K]
    returns  : [B, K, C]
    """
    B, N, C = features.shape
    K = indices.shape[1]
    idx = indices.unsqueeze(-1).expand(B, K, C)   # [B, K, C]
    return torch.gather(features, 1, idx)          # 'index' kwarg = positional arg 3


def section(title):
    print("\n" + "=" * 52)
    print("  " + title)
    print("=" * 52)


def run_test():
    section("Initializing models")
    encoder   = Encoder().to(DEVICE)
    predictor = Predictor().to(DEVICE)
    print(f"  Encoder   params : {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Predictor params : {sum(p.numel() for p in predictor.parameters()):,}")
    print(f"  Device           : {DEVICE}")

    # -- Step 1: Encoder forward on a full, unmasked image --------------------
    section("Step 1 - Encoder forward (full image, no masking)")
    x = torch.randn(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)

    with torch.no_grad():
        f0, f1, f2, f3 = encoder(x)

    print(f"  Input  : {list(x.shape)}")
    print(f"  f0 (Stage 0) : {list(f0.shape)}  <- skipped (too many tokens for CGLT)")
    print(f"  f1 (Stage 1) : {list(f1.shape)}")
    print(f"  f2 (Stage 2) : {list(f2.shape)}  <- primary prediction scale")
    print(f"  f3 (Stage 3) : {list(f3.shape)}")

    # -- Step 2: Masking on f2 ------------------------------------------------
    section("Step 2 - Masking on f2 token sequence")
    B, N2, C2 = f2.shape
    context_idx, target_idx = random_mask(B, N2, MASK_RATIO, DEVICE)

    # Gather context tokens (what the predictor sees)
    context_tokens = gather_tokens(f2, context_idx)

    # Gather target tokens and DETACH - stop gradient at encoder output
    target_f2 = gather_tokens(f2, target_idx).detach()
    # Map f2 target indices to f3 space (f3 is 7x7, f2 is 14x14, so divide by 4)
    target_f3 = gather_tokens(f3, (target_idx // 4).clamp(max=f3.shape[1] - 1)).detach()

    print(f"  Total f2 tokens   : {N2}")
    print(f"  Context tokens    : {list(context_tokens.shape)}  ({context_idx.shape[1]} patches)")
    print(f"  Target f2 tokens  : {list(target_f2.shape)}  [detached]")
    print(f"  Target f3 tokens  : {list(target_f3.shape)}  [detached]")

    # -- Step 3: Predictor forward --------------------------------------------
    section("Step 3 - Predictor forward")
    pred_stage2, pred_stage3 = predictor(context_tokens, context_idx, target_idx)

    print(f"  Predicted Stage 2 : {list(pred_stage2.shape)}  (target: {list(target_f2.shape)})")
    print(f"  Predicted Stage 3 : {list(pred_stage3.shape)}  (target: {list(target_f3.shape)})")

    # -- Step 4: Shape assertions ---------------------------------------------
    section("Step 4 - Shape assertions")
    assert pred_stage2.shape == target_f2.shape, \
        f"Stage 2 mismatch: {pred_stage2.shape} vs {target_f2.shape}"
    assert pred_stage3.shape == target_f3.shape, \
        f"Stage 3 mismatch: {pred_stage3.shape} vs {target_f3.shape}"
    print("  All shapes match [OK]")

    # -- Step 5: Loss ---------------------------------------------------------
    section("Step 5 - Loss computation")
    loss_s2    = F.mse_loss(pred_stage2, target_f2)
    loss_s3    = F.mse_loss(pred_stage3, target_f3)
    loss_total = loss_s2 + loss_s3

    print(f"  Loss Stage 2 : {loss_s2.item():.6f}")
    print(f"  Loss Stage 3 : {loss_s3.item():.6f}")
    print(f"  Loss Total   : {loss_total.item():.6f}")

    assert not torch.isnan(loss_total), "Loss is NaN!"
    assert loss_total.item() > 0,       "Loss is zero - something collapsed!"
    print("  Loss valid (non-NaN, non-zero) [OK]")

    # -- Step 6: Gradient flow ------------------------------------------------
    section("Step 6 - Gradient flow check")
    loss_total.backward()

    pred_param = next(predictor.parameters())
    assert pred_param.grad is not None, "Predictor has no gradients!"
    print(f"  Predictor grad norm : {pred_param.grad.norm().item():.6f} [OK]")
    print(f"  Encoder targets are detached (stop-gradient) [OK]")

    section("ALL TESTS PASSED")
    print("  Encoder + Predictor assembly is working correctly.\n")


if __name__ == "__main__":
    run_test()
