import json
import os
import cv2
import numpy as np
from pathlib import Path

def extract_coronary_segmentations_from_syntax(split='train'):
    """
    Extrage segmentările de coronariene din dataset-ul syntax.
    Pacienții 1001-2000 din syntax.
    """
    print(f"\n--- Extracting Coronary Segmentations from Syntax ({split}) ---")
    
    # Paths
    json_path = rf'D:\Collateral Coronary Vessels XAI\data\ARCADE\RAW\arcade\syntax\{split}\annotations\{split}.json'
    output_dir = rf'D:\Collateral Coronary Vessels XAI\data\ARCADE\{split}\labels'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data['images'])} images from syntax...")
    
    generated_files = {}
    
    for img_info in data['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        h, w = img_info['height'], img_info['width']
        
        # Get patient number from filename (e.g., "1.png" -> 1)
        patient_num = int(Path(file_name).stem)
        
        # Get all annotations for this image
        anns = [ann for ann in data['annotations'] if ann['image_id'] == img_id]
        
        if len(anns) == 0:
            continue
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw all vessel segments (all categories in syntax are vessels)
        for ann in anns:
            if 'segmentation' in ann:
                for poly in ann['segmentation']:
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 255)
        
        # Save mask
        output_path = os.path.join(output_dir, f'label_coronary_{patient_num}.png')
        cv2.imwrite(output_path, mask)
        
        # Store relative path for dataset.json
        relative_path = f"src\\data\\processed\\ARCADE\\{split}\\labels\\label_coronary_{patient_num}.png"
        generated_files[patient_num] = relative_path
    
    print(f"Generated {len(generated_files)} coronary label files for {split} (syntax)")
    return generated_files


def extract_coronary_segmentations_from_stenosis(split='train'):
    """
    Extrage segmentările de coronariene din dataset-ul stenosis.
    Pacienții 1-1000, doar categoriile 1-25 (NU stenoza care e categoria 26).
    """
    print(f"\n--- Extracting Coronary Segmentations from Stenosis ({split}) ---")
    
    # Paths
    json_path = rf'D:\Collateral Coronary Vessels XAI\data\ARCADE\RAW\arcade\stenosis\{split}\annotations\{split}.json'
    output_dir = rf'D:\Collateral Coronary Vessels XAI\data\ARCADE\{split}\labels'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data['images'])} images from stenosis...")
    
    generated_files = {}
    
    for img_info in data['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        h, w = img_info['height'], img_info['width']
        
        # Get patient number from filename
        patient_num = int(Path(file_name).stem)
        
        # Get all annotations for this image
        anns = [ann for ann in data['annotations'] if ann['image_id'] == img_id]
        
        # Filter ONLY coronary vessels (categories 1-25), NOT stenosis (26)
        coronary_anns = [ann for ann in anns if ann['category_id'] < 26]
        
        if len(coronary_anns) == 0:
            continue
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw vessel segments
        for ann in coronary_anns:
            if 'segmentation' in ann:
                for poly in ann['segmentation']:
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 255)
        
        # Save mask
        output_path = os.path.join(output_dir, f'label_coronary_{patient_num}.png')
        cv2.imwrite(output_path, mask)
        
        # Store relative path for dataset.json
        relative_path = f"src\\data\\processed\\ARCADE\\{split}\\labels\\label_coronary_{patient_num}.png"
        generated_files[patient_num] = relative_path
    
    print(f"Generated {len(generated_files)} coronary label files for {split} (stenosis)")
    return generated_files


def update_dataset_json(coronary_files_dict):
    """
    Actualizează dataset.json cu paths către segmentările coronariene.
    coronary_files_dict = {
        'train': {patient_num: path, ...},
        'test': {patient_num: path, ...},
        'val': {patient_num: path, ...}
    }
    """
    print("\n--- Updating dataset.json ---")
    
    dataset_path = r'D:\Collateral Coronary Vessels XAI\data\ARCADE\dataset.json'
    
    # Load dataset.json
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Update each split
    for split in ['train', 'test', 'val']:
        if split not in dataset:
            continue
        
        coronary_files = coronary_files_dict.get(split, {})
        
        # Iterate through all patients in this split
        for patient_key, patient_data in dataset[split].items():
            # Extract patient number from key (e.g., "pacient_1" -> 1)
            patient_num = int(patient_key.split('_')[1])
            
            # Add coronary_label field
            if patient_num in coronary_files:
                patient_data['coronary_label'] = coronary_files[patient_num]
            else:
                patient_data['coronary_label'] = None
    
    # Save updated dataset.json
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    
    print(f"✓ Updated dataset.json with coronary_label paths")


def main():
    """
    Main function to generate all coronary segmentations and update dataset.json
    """
    print("=" * 60)
    print("Generating Coronary Vessel Segmentations")
    print("=" * 60)
    
    all_coronary_files = {}
    
    # Process all splits
    for split in ['train', 'test', 'val']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}")
        
        # Extract from syntax (patients 1001-2000)
        syntax_files = extract_coronary_segmentations_from_syntax(split)
        
        # Extract from stenosis (patients 1-1000, only coronaries, not stenosis)
        stenosis_files = extract_coronary_segmentations_from_stenosis(split)
        
        # Combine (should not overlap as syntax=1001-2000, stenosis=1-1000)
        combined_files = {**stenosis_files, **syntax_files}
        all_coronary_files[split] = combined_files
        
        print(f"\nTotal for {split}: {len(combined_files)} coronary label files")
    
    # Update dataset.json
    update_dataset_json(all_coronary_files)
    
    print("\n" + "=" * 60)
    print("✓ All coronary segmentations generated successfully!")
    print("=" * 60)
    
    # Summary
    total = sum(len(files) for files in all_coronary_files.values())
    print(f"\nSummary:")
    print(f"  Train: {len(all_coronary_files.get('train', {}))} files")
    print(f"  Test: {len(all_coronary_files.get('test', {}))} files")
    print(f"  Val: {len(all_coronary_files.get('val', {}))} files")
    print(f"  Total: {total} coronary label files")


if __name__ == "__main__":
    main()
