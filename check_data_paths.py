#!/usr/bin/env python3
"""Check where data files actually exist"""
import os
import json

# Check the JSON file
json_path = '/root/Collateral-Coronary-Vessels-XAI/data/ARCADE/processed/dataset.json'

print("=" * 80)
print("DATA PATH DIAGNOSTIC")
print("=" * 80)

# Load JSON and check a few sample paths
with open(json_path, 'r') as f:
    data = json.load(f)

# Get some sample paths from the 'extra' section
sample_paths = []
if 'extra' in data and 'pretrain' in data['extra']:
    for key, item in list(data['extra']['pretrain'].items())[:5]:
        sample_paths.append(item['data'])

print(f"\nFound {len(sample_paths)} sample paths from 'extra.pretrain'\n")

# Check each path in multiple locations
for path_str in sample_paths[:3]:  # Check first 3
    print(f"Path in JSON: {path_str}")
    print("-" * 40)
    
    # Clean path
    clean_path = path_str.replace('\\', '/')
    
    # Try different locations
    locations = [
        f"/root/Collateral-Coronary-Vessels-XAI/{clean_path}",
        f"/workspace/Collateral-Coronary-Vessels-XAI/{clean_path}",
        f"/root/Collateral-Coronary-Vessels-XAI/data/ARCADE/{clean_path}",
        f"/workspace/Collateral-Coronary-Vessels-XAI/data/ARCADE/{clean_path}",
    ]
    
    # Also extract just the filename and search for it
    if 'collateral coronary vessels' in clean_path.lower():
        idx = clean_path.lower().find('collateral coronary vessels')
        after_project = clean_path[clean_path.find('/', idx)+1:]
        locations.append(f"/root/Collateral-Coronary-Vessels-XAI/{after_project}")
        locations.append(f"/workspace/Collateral-Coronary-Vessels-XAI/{after_project}")
    
    found = False
    for loc in locations:
        exists = os.path.exists(loc)
        print(f"  {'✓' if exists else '✗'} {loc}")
        if exists:
            found = True
            break
    
    if not found:
        print(f"  ⚠ FILE NOT FOUND IN ANY LOCATION!")
    print()

# Check if symlinks exist
print("\nChecking symlinks:")
print("-" * 80)
extra_link = '/root/Collateral-Coronary-Vessels-XAI/data/Extra'
processed_link = '/root/Collateral-Coronary-Vessels-XAI/data/ARCADE/processed'

for link in [extra_link, processed_link]:
    if os.path.islink(link):
        target = os.readlink(link)
        print(f"✓ Symlink exists: {link} -> {target}")
    elif os.path.exists(link):
        print(f"✓ Directory exists (not symlink): {link}")
    else:
        print(f"✗ Does not exist: {link}")

print("\n" + "=" * 80)
