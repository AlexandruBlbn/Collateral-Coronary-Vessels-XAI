from helpers import *
import sys
sys.path.append(".")
from skimage.filters import gaussian, meijering, sato
from skimage.color import rgb2gray
from skimage.exposure import equalize_adapthist
from skimage.restoration import denoise_nl_means
from skimage.morphology import opening, disk
from skimage.morphology import skeletonize
from data import data as data
from skimage.measure import label, regionprops



datas = data.finetune_dataset()

def remove_objs(img, min_size=20):
    img = np.array(img)
    binary = img > 0.01
    labeled = label(binary)
    cleand = np.zeros_like(img)
    for region in regionprops(labeled):
        if region.area >= min_size:
            cleand[labeled == region.label] = img[labeled == region.label]
    return cleand

def get_fov(image, threshold=10, erosion_size = 10):
    '''
    Gets fov for the image, by thresholding and finding the largest connected component, then eroding it.
    args:
        image: np.array, the image to get fov for
        threshold: int, the threshold to use for binarization
        erosion_size: int, the size of the erosion kernel
    '''
    
    
    binary = (image>threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    fov_mask = (labels==largest).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size*2+1, erosion_size*2+1))
    fov_mask = cv2.erode(fov_mask, kernel)
    return fov_mask

    


image, gt_label = datas[50]
image = rgb2gray(np.array(image))
image_original = image.copy()
# image = equalize_adapthist(image, clip_limit=0.01)
image = denoise_nl_means(image, h=0.020, fast_mode=True)
 # image = gaussian(image, sigma=1)
sato_img = sato(image, (1,4))
sato_img[sato_img < 0.01] = 0
meijering_img = meijering(image, (1,4))     
meijering_img[meijering_img < 0.05] = 0  
sato_img = opening(sato_img, disk(2))
meijering_img = opening(meijering_img, disk(2))  
sato_bin = sato_img > 0.01
meijering_bin = meijering_img > 0.01    
fov = get_fov((image*255).astype(np.uint8), threshold=30, erosion_size=20)
sato_bin = sato_bin * fov
meijering_bin = meijering_bin * fov
consensus = sato_bin & meijering_bin
consensus = consensus.astype(float)
consensus = remove_objs(consensus, min_size=500)
skeleton_sato = skeletonize(sato_img)
skeleton_mej = skeletonize(meijering_img)
       

plt.figure(1, figsize=(10, 10))
plt.subplot(1, 6, 1)
plt.imshow(image_original, cmap='gray')
plt.subplot(1, 6, 2)
plt.imshow(sato_img, cmap='gray')
plt.subplot(1, 6, 3)
plt.imshow(meijering_img, cmap='gray')
plt.subplot(1, 6, 4)
plt.imshow(skeleton_sato, cmap='gray')
plt.subplot(1, 6, 5)
plt.imshow(skeleton_mej, cmap='gray')
plt.subplot(1, 6, 6)
plt.imshow(consensus, cmap='gray')
plt.savefig("vessels_prior.png", dpi=300)

 


