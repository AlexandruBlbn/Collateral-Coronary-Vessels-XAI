import cv2, numpy as np
from data.vessel_consensus import hessian_eigenvalue_ratio, structure_tensor_coherence, scale_consistency

img = cv2.imread("data/extra/trainB/0.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (512, 512))
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_c = clahe.apply(img)

s1 = hessian_eigenvalue_ratio(img_c)
s2 = structure_tensor_coherence(img_c)
s3 = scale_consistency(img_c)

pct = 95
t1 = s1 > np.percentile(s1, pct)
t2 = s2 > np.percentile(s2, pct)
t3 = s3 > np.percentile(s3, pct)

print(f"Each >{pct}pct: s1={t1.sum()}, s2={t2.sum()}, s3={t3.sum()}")
overlap3 = t1 & t2 & t3
print(f"3-way overlap: {overlap3.sum()}")

# Try a looser threshold
for p in [85, 80, 75, 70]:
    t1 = s1 > np.percentile(s1, p)
    t2 = s2 > np.percentile(s2, p)
    t3 = s3 > np.percentile(s3, p)
    o = t1 & t2 & t3
    print(f"  pct={p}: 3-way overlap={o.sum()} ({100*o.sum()/s1.size:.2f}%)")

# Check: do top-80 hessian and top-80 coherence mostly agree on vessels?
t1_80 = s1 > np.percentile(s1, 80)
t2_80 = s2 > np.percentile(s2, 80)
t3_80 = s3 > np.percentile(s3, 80)
print(f"\n>80pct each: s1={t1_80.sum()}, s2={t2_80.sum()}, s3={t3_80.sum()}")
o = t1_80 & t2_80 & t3_80
print(f"3-way@80: {o.sum()} ({100*o.sum()/s1.size:.2f}%)")

# Save the 3-way binary for visual inspection
cv2.imwrite("data/_consensus_3way_80.png", (o.astype(np.uint8) * 255))
print("Saved data/_consensus_3way_80.png")
