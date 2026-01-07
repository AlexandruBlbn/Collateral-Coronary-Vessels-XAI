import torch
import numpy as np
from torch.utils.data import Dataset
from skimage.filters import frangi, median, threshold_otsu
from skimage import exposure
from skimage.measure import label, regionprops, block_reduce
from skimage.morphology import disk, reconstruction, white_tophat, remove_small_objects, dilation, binary_opening, binary_erosion
from skimage.exposure import rescale_intensity
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt

class VasoMIMMaskGenerator:
    def __init__(self, input_size=256, mask_patch_size=32, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio
        
        self.grid_size = self.input_size // self.mask_patch_size
        self.token_count = self.grid_size ** 2
        self.mask_count = int(np.ceil(self.token_count * mask_ratio))

    def _create_fov_mask(self, img, intensity_threshold=0.08, erosion_size=12):
        content_mask = img > intensity_threshold
        
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(img, size=15)
        local_sq_mean = uniform_filter(img**2, size=15)
        local_var = local_sq_mean - local_mean**2
        local_var = np.clip(local_var, 0, None)
        
        variance_mask = local_var > 0.001
        fov_mask = content_mask & variance_mask
        fov_mask = ndi.binary_fill_holes(fov_mask)
        
        if erosion_size > 0:
            fov_mask = binary_erosion(fov_mask, disk(erosion_size))
        
        labeled_fov, num_fov = ndi.label(fov_mask)
        if num_fov > 0:
            counts = np.bincount(labeled_fov.flat)
            if len(counts) > 1:
                largest_label = np.argmax(counts[1:]) + 1
                fov_mask = (labeled_fov == largest_label)
        
        return fov_mask

    def _keep_largest_vessel_tree(self, mask):
        mask = remove_small_objects(mask, min_size=30)
        labels = label(mask)
        if labels.max() == 0: return mask
        
        regions = regionprops(labels)
        regions.sort(key=lambda x: x.area, reverse=True)
        
        output = np.zeros_like(mask, dtype=bool)
        if len(regions) > 0:
            output[labels == regions[0].label] = True
        
        for i in range(1, min(len(regions), 4)):
            if regions[i].area > regions[0].area * 0.10:
                output[labels == regions[i].label] = True
                
        return output

    def _filter_linear_artifacts(self, mask):
        labels = label(mask)
        for region in regionprops(labels):
            if region.eccentricity > 0.990 and region.area > 50:
                if region.minor_axis_length < 4:
                    mask[labels == region.label] = False
        return mask

    def _remove_thick_bones(self, mask, thickness_threshold=6):
        thick_structures = binary_opening(mask, disk(thickness_threshold))
        thick_structures = dilation(thick_structures, disk(2))
        mask_clean = mask & ~thick_structures
        return mask_clean

    def _detect_and_remove_ribs(self, mask, img_original, angle_tolerance=70):
        edges = canny(mask.astype(float), sigma=1)
        try:
            lines = probabilistic_hough_line(edges, threshold=40, line_length=25, line_gap=5)
        except:
            return mask
        
        if not lines:
            return mask
        
        rib_mask = np.zeros_like(mask, dtype=bool)
        for p0, p1 in lines:
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            if dx == 0: angle = 90
            else: angle = np.abs(np.degrees(np.arctan2(dy, dx)))
            
            is_horizontal = angle < angle_tolerance or angle > (180 - angle_tolerance)
            x_center = (p0[0] + p1[0]) / 2
            is_lateral = x_center < mask.shape[1] * 0.30 or x_center > mask.shape[1] * 0.70
            line_length = np.sqrt(dx**2 + dy**2)
            is_long = line_length > 20
            
            if is_horizontal and is_lateral and is_long:
                from skimage.draw import line
                rr, cc = line(p0[1], p0[0], p1[1], p1[0])
                valid = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
                rib_mask[rr[valid], cc[valid]] = True
        
        if rib_mask.any():
            rib_mask = dilation(rib_mask, disk(4))
        
        return mask & ~rib_mask

    def _verify_vessel_intensity(self, mask, img_original, min_contrast=0.03):
        labels = label(mask)
        verified_mask = np.zeros_like(mask, dtype=bool)
        
        for region in regionprops(labels, intensity_image=img_original):
            region_mask = labels == region.label
            dilated = dilation(region_mask, disk(4))
            neighborhood = dilated & ~region_mask
            
            if neighborhood.sum() > 10:
                vessel_intensity = img_original[region_mask].mean()
                background_intensity = img_original[neighborhood].mean()
                contrast = background_intensity - vessel_intensity
                
                if contrast > min_contrast:
                    verified_mask[region_mask] = True
                elif contrast > -0.02:
                    if region.area < 200:
                        verified_mask[region_mask] = True
        return verified_mask

    def get_vasomim_probability_map(self, img_np):
        # 1. Pre-procesare
        img_denoised = median(img_np, disk(1))
        img_inv = 1.0 - img_denoised
        img_tophat = white_tophat(img_inv, disk(10))
        
        p2, p99 = np.percentile(img_tophat, (2, 99.5))
        img_rescaled = rescale_intensity(img_tophat, in_range=(p2, p99))
        img_clahe = exposure.equalize_adapthist(img_rescaled, clip_limit=0.02, kernel_size=(16, 16))
        img_final_pre = img_clahe
        
        fov_mask = self._create_fov_mask(img_np, intensity_threshold=0.08, erosion_size=12)

        # 2. Frangi Multi-scale
        scales_config = [
            {'sigma': 2.5, 'thresh_factor': 0.8}, 
            {'sigma': 1.0, 'thresh_factor': 0.3}
        ]
        
        final_mask = np.zeros_like(img_np, dtype=bool)
        combined_frangi_map = np.zeros_like(img_np, dtype=float)
        
        for i, config in enumerate(scales_config):
            f_map = frangi(img_final_pre, sigmas=[config['sigma']], 
                           black_ridges=False, alpha=0.5, beta=0.5, gamma=15)
            f_map = f_map * fov_mask.astype(float)
            
            if f_map.max() > 0: f_map /= f_map.max()
            combined_frangi_map = np.maximum(combined_frangi_map, f_map)
            
            thresh = threshold_otsu(f_map[f_map > 0]) if (f_map > 0).any() else 0.1
            mask_scale = f_map > (thresh * config['thresh_factor'])
            
            if i == 0:
                mask_scale = self._keep_largest_vessel_tree(mask_scale)
                final_mask = mask_scale
            else:
                seed = dilation(final_mask, disk(3))
                intersection = seed & mask_scale
                if intersection.any():
                    mask_connected = reconstruction(intersection, mask_scale, method='dilation')
                    final_mask = final_mask | mask_connected.astype(bool)

        # 3. Post-procesare
        final_mask = self._remove_thick_bones(final_mask, thickness_threshold=7)
        final_mask = self._verify_vessel_intensity(final_mask, img_np, min_contrast=0.02)
        final_mask = self._detect_and_remove_ribs(final_mask, img_np, angle_tolerance=25)
        final_mask = self._filter_linear_artifacts(final_mask)
        final_mask = self._keep_largest_vessel_tree(final_mask)

        # 4. Generare Mapă
        pixel_prob_map = combined_frangi_map.copy()
        if pixel_prob_map.max() > 0:
            pixel_prob_map = pixel_prob_map / pixel_prob_map.max()
        pixel_prob_map = pixel_prob_map * final_mask.astype(float)

        # Filtrare coerență spațială
        labeled_pixels, num_components = ndi.label(final_mask)
        if num_components > 0:
            component_intensities = []
            for comp_label in range(1, num_components + 1):
                comp_mask = (labeled_pixels == comp_label)
                comp_intensity = pixel_prob_map[comp_mask].sum()
                component_intensities.append((comp_label, comp_intensity, comp_mask))
            
            component_intensities.sort(key=lambda x: x[1], reverse=True)
            if len(component_intensities) > 0:
                main_intensity = component_intensities[0][1]
                total_intensity = sum(c[1] for c in component_intensities)
                kept_pixel_mask = np.zeros_like(final_mask, dtype=bool)
                for comp_label, comp_int, comp_mask in component_intensities:
                    if comp_int >= main_intensity * 0.15 or comp_int >= total_intensity * 0.1:
                        kept_pixel_mask |= comp_mask
                pixel_prob_map = pixel_prob_map * kept_pixel_mask.astype(float)

        # Prior anatomic
        h_px, w_px = pixel_prob_map.shape
        y_px, x_px = np.ogrid[:h_px, :w_px]
        center_y_px, center_x_px = h_px / 2, w_px / 2
        dist_px = np.sqrt((x_px - center_x_px)**2 + (y_px - center_y_px)**2)
        max_dist_px = np.sqrt(center_x_px**2 + center_y_px**2)
        dist_norm_px = dist_px / max_dist_px
        sigma_center = 0.6
        center_weight = np.exp(-0.5 * (dist_norm_px / sigma_center) ** 2)
        center_weight = 0.3 + 0.7 * center_weight
        pixel_prob_map = pixel_prob_map * center_weight
        
        # Reducere la patch-uri
        vasomim_prob_map = block_reduce(pixel_prob_map, block_size=(self.mask_patch_size, self.mask_patch_size), func=np.mean)
        
        if vasomim_prob_map.sum() > 0:
            vasomim_prob_map = vasomim_prob_map / vasomim_prob_map.sum()
            
        return vasomim_prob_map, final_mask

    def __call__(self, img):
        # img: Tensor (C, H, W) sau (H, W) sau numpy
        if isinstance(img, torch.Tensor):
            img_np = img.squeeze().cpu().numpy()
        else:
            img_np = img
            
        prob_map, anatomy_mask = self.get_vasomim_probability_map(img_np)
        flat_prob = prob_map.flatten()
        
        if flat_prob.sum() == 0:
            flat_prob = np.ones_like(flat_prob) / len(flat_prob)
        else:
            flat_prob = flat_prob / flat_prob.sum()
            
        mask_idx = np.random.choice(self.token_count, size=self.mask_count, replace=False, p=flat_prob)
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.grid_size, self.grid_size))
        mask = mask.repeat(self.mask_patch_size, axis=0).repeat(self.mask_patch_size, axis=1)
        
        return torch.from_numpy(mask), torch.from_numpy(anatomy_mask)

class ArcadeDatasetVasoMIM(Dataset):
    def __init__(self, arcade_dataset, mask_generator):
        self.arcade_dataset = arcade_dataset
        self.mask_generator = mask_generator
        
    def __len__(self):
        return len(self.arcade_dataset)
    
    def __getitem__(self, idx):
        image = self.arcade_dataset[idx]
        mask, anatomy_mask = self.mask_generator(image)
        return image, mask, anatomy_mask