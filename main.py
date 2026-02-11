import json

with open('data/ARCADE/processed/dataset.json', 'r') as f:
    data = json.load(f)

# Count DataValidation samples
dataval = data['DataValidation']
positive_samples = []
negative_samples = []

for key, item in dataval.items():
    if key.startswith('positive_'):
        positive_samples.append(item['data'])
    elif key.startswith('negative_'):
        negative_samples.append(item['data'])

num_pos = len(positive_samples)
num_neg = len(negative_samples)

train_pos_count = int(num_pos * 0.10)
train_neg_count = int(num_neg * 0.10)

print(f"DataValidation total: {num_pos} positives, {num_neg} negatives")
print(f"\nTraining set composition:")
print(f"  - Stenoza train: 1000 (class 1)")
print(f"  - Extra/pretrain: {len(data['extra']['pretrain'])} (class 0)")
print(f"  - DataValidation 10%: {train_pos_count} positives (class 1), {train_neg_count} negatives (class 0)")
print(f"\n  Total class 1: {1000 + train_pos_count}")
print(f"  Total class 0: {len(data['extra']['pretrain']) + train_neg_count}")
print(f"  Ratio class0/class1: {(len(data['extra']['pretrain']) + train_neg_count) / (1000 + train_pos_count):.2f}")