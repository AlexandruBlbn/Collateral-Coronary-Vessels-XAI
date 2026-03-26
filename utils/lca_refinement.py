import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None
    print("segmentation_models_pytorch is not installed. LCARefinementUNet will not be available.")

# 1. The Secondary Model: Cascade / Refinement Network
class LCARefinementUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", num_classes=1):
        super().__init__()
        if smp is None:
            raise ImportError("Please install segmentation_models_pytorch to use LCARefinementUNet")
            
        # Note: in_channels=2 (Image + Initial Mask)
        self.model = smp.Unet(
            encoder_name=encoder_name,
            in_channels=2, 
            classes=num_classes,
        )

    def forward(self, image, initial_mask):
        # image: [B, 1, H, W] or [B, 3, H, W]
        # initial_mask: [B, 1, H, W] (probabilities or binary mask)
        x = torch.cat([image, initial_mask], dim=1)
        return self.model(x)


# 2. Topological Post-Processing (Classic Computer Vision)
def refine_lca_mask(binary_mask: np.ndarray, min_area: int = 150, close_kernel_size: int = 5) -> np.ndarray:
    """
    1. Morphological Close: Bridges small 1-2 pixel gaps where the model disconnected a vessel.
    2. Connected Components: Deletes floating false-positive "vessels" in the background.
    """
    out = (binary_mask > 0).astype(np.uint8)

    # 1. Morphological Closing to bridge micro-disconnections
    if close_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    # 2. Connected Component Analysis to remove small isolated blobs
    if min_area > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        cleaned = np.zeros_like(out)
        
        # stats[label, cv2.CC_STAT_AREA] gives the pixel count of that component
        # start from 1 to ignore the background (label 0)
        for label_id in range(1, n_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area >= min_area:
                cleaned[labels == label_id] = 1
                
        out = cleaned

    return out


# 3. Mask Validation using Frangi Filter
def frangi_suppression(pred_mask: np.ndarray, original_img: np.ndarray, frangi_filter_func) -> np.ndarray:
    """
    Suppresses network predictions where Frangi response is near zero.
    Requires a frangi_filter_func that takes an image and returns a normalized response (0 to 1).
    """
    # 1. Get raw Frangi vesselness (normalized 0 to 1)
    frangi_response = frangi_filter_func(original_img)
    
    # 2. Suppress network predictions where Frangi response is near zero 
    # (meaning it's definitely not a tubular structure)
    frangi_mask = (frangi_response > 0.05).astype(np.uint8)
    
    # 3. Logical AND: It must be predicted by the network AND look somewhat like a tube
    refined_mask = np.logical_and(pred_mask, frangi_mask).astype(np.uint8)
    
    return refine_lca_mask(refined_mask)


# 4. Upstream Fix: The clDice (Centerline Dice) Loss
def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
        p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
        p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
        return torch.min(torch.min(p1, p2), p3)

def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3,3,3), (1,1,1), (1,1,1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iter_=10):
    img1  =  soft_open(img)
    skel  =  F.relu(img-img1)
    for j in range(iter_):
        img  =  soft_erode(img)
        img1  =  soft_open(img)
        delta  =  F.relu(img-img1)
        skel  =  skel +  F.relu(delta-skel)
    return skel

class soft_cldice(nn.Module):
    def __init__(self, iter_=10, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        
        # Calculate precision and sensitivity for centerline dice
        # Note: assuming shape is [B, C, H, W]
        tprec = (torch.sum(torch.mul(skel_pred, y_true)[:,0:,...])+self.smooth)/(torch.sum(skel_pred[:,0:,...])+self.smooth)    
        tsens = (torch.sum(torch.mul(skel_true, y_pred)[:,0:,...])+self.smooth)/(torch.sum(skel_true[:,0:,...])+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice

def lca_combined_loss(y_pred, y_true, bce_weight=0.5, tversky_weight=0.2, cldice_weight=0.3):
    '''
    Example of a combined loss using BCE, Tversky/Dice, and clDice.
    '''
    bce = nn.BCEWithLogitsLoss()(y_pred, y_true)
    
    # Simple Dice implementation for Tversky replacement
    pred_sig = torch.sigmoid(y_pred)
    intersection = (pred_sig * y_true).sum()
    dice = 1 - (2. * intersection + 1) / (pred_sig.sum() + y_true.sum() + 1)
    
    cldice = soft_cldice()(y_true, pred_sig)
    
    return bce_weight * bce + tversky_weight * dice + cldice_weight * cldice
