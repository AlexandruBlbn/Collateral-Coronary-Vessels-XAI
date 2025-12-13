import json
import os
import cv2
import numpy as np

def create_combined_segmentation_patient_1():
    """
    Creează segmentarea combinată pentru pacientul 1:
    - Imaginea este din RAW/arcade/stenosis
    - Vasele coronariene din syntax (alb)
    - Stenoza din stenosis (roșu)
    - Generează 2 imagini: segmentare pură și overlay
    """
    print("\n--- Creating Combined Segmentation for Patient 1 ---")
    
    # Paths
    syntax_json_path = r'D:\Collateral Coronary Vessels XAI\data\ARCADE\RAW\arcade\syntax\train\annotations\train.json'
    stenosis_json_path = r'D:\Collateral Coronary Vessels XAI\data\ARCADE\RAW\arcade\stenosis\train\annotations\train.json'
    stenosis_img_dir = r'D:\Collateral Coronary Vessels XAI\data\ARCADE\RAW\arcade\stenosis\train\images'
    debug_dir = r'D:\Collateral Coronary Vessels XAI\debug_patient_1'
    os.makedirs(debug_dir, exist_ok=True)
    
    target_file = "1.png"
    
    # Load JSONs
    print("Loading JSONs...")
    with open(syntax_json_path, 'r') as f:
        syntax_data = json.load(f)
    with open(stenosis_json_path, 'r') as f:
        stenosis_data = json.load(f)
    
    # Find image IDs
    syntax_img_id = None
    for img in syntax_data['images']:
        if img['file_name'] == target_file:
            syntax_img_id = img['id']
            break
    
    stenosis_img_id = None
    for img in stenosis_data['images']:
        if img['file_name'] == target_file:
            stenosis_img_id = img['id']
            break
    
    if syntax_img_id is None or stenosis_img_id is None:
        print(f"Could not find {target_file} in both datasets")
        return
    
    print(f"Found {target_file}: Syntax ID={syntax_img_id}, Stenosis ID={stenosis_img_id}")
    
    # Get annotations
    syntax_anns = [ann for ann in syntax_data['annotations'] if ann['image_id'] == syntax_img_id]
    stenosis_anns = [ann for ann in stenosis_data['annotations'] if ann['image_id'] == stenosis_img_id]
    
    print(f"Found {len(syntax_anns)} vessel segments (Syntax)")
    print(f"Found {len(stenosis_anns)} stenosis segments (Stenosis)")
    
    # Load original image from STENOSIS folder
    img_path = os.path.join(stenosis_img_dir, target_file)
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Failed to load: {img_path}")
        return
    
    h, w = original_img.shape[:2]
    
    # Create masks
    vessels_mask = np.zeros((h, w), dtype=np.uint8)
    stenosis_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw vessels from Syntax annotations
    for ann in syntax_anns:
        if 'segmentation' in ann:
            for poly in ann['segmentation']:
                pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(vessels_mask, [pts], 255)
    
    # Draw stenosis from Stenosis annotations
    for ann in stenosis_anns:
        if 'segmentation' in ann:
            for poly in ann['segmentation']:
                pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(stenosis_mask, [pts], 255)
    
    # Create colored segmentation map (3-channel)
    segmentation_colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Vessels in white (255, 255, 255)
    segmentation_colored[vessels_mask == 255] = [255, 255, 255]
    
    # Stenosis in red (0, 0, 255) in BGR
    segmentation_colored[stenosis_mask == 255] = [0, 0, 255]
    
    # Save segmentation image
    seg_output_path = os.path.join(debug_dir, 'patient_1_segmentation.png')
    cv2.imwrite(seg_output_path, segmentation_colored)
    print(f"Saved segmentation to: {seg_output_path}")
    
    # Create overlay on original image
    overlay = original_img.copy()
    
    # Apply vessels in white
    overlay[vessels_mask == 255] = [255, 255, 255]
    
    # Apply stenosis in red
    overlay[stenosis_mask == 255] = [0, 0, 255]
    
    # Blend with original (60% original, 40% overlay)
    blended = cv2.addWeighted(original_img, 0.6, overlay, 0.4, 0)
    
    # Save overlay image
    overlay_output_path = os.path.join(debug_dir, 'patient_1_overlay.png')
    cv2.imwrite(overlay_output_path, blended)
    print(f"Saved overlay to: {overlay_output_path}")
    
    # Also save the original image from stenosis folder for reference
    original_output_path = os.path.join(debug_dir, 'patient_1_original_from_stenosis.png')
    cv2.imwrite(original_output_path, original_img)
    print(f"Saved original image to: {original_output_path}")
    
    print("\n✓ Done! Check the debug_patient_1 folder for:")
    print("  - patient_1_segmentation.png (segmentation only)")
    print("  - patient_1_overlay.png (overlay on original)")
    print("  - patient_1_original_from_stenosis.png (original image)")

if __name__ == "__main__":
    create_combined_segmentation_patient_1()
