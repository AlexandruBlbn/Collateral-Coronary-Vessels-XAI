import json

dataset_path = 'D:\\Collateral Coronary Vessels XAI\\data\\ARCADE\\dataset.json'

with open(dataset_path, 'r') as f:
    data = json.load(f)

print("=" * 60)
print("DATASET ANALYSIS")
print("=" * 60)

for split in ['train', 'validation', 'test']:
    if split not in data:
        print(f"\n⚠ Split '{split}' nu există în dataset.json")
        continue
    
    all_samples = list(data[split].values())
    with_coronary = [s for s in all_samples if s.get('coronary_label') is not None]
    
    print(f"\n{split.upper()}:")
    print(f"  Total pacienți: {len(all_samples)}")
    print(f"  Cu coronary_label: {len(with_coronary)}")
    print(f"  Fără coronary_label: {len(all_samples) - len(with_coronary)}")
    
    if len(with_coronary) > 0:
        # Afișează primii 3 cu coronary_label
        print(f"  Primii 3 cu coronary_label:")
        for i, sample in enumerate(with_coronary[:3]):
            print(f"    {i+1}. coronary_label: {sample.get('coronary_label')}")

print("\n" + "=" * 60)
