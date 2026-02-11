import torchvision.transforms as transforms
from PIL import Image

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4), local_crops_number=4):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        # Transformare Globală (Mare - 224px sau cât ai tu input_size)
        self.global_transfo = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Transformare Locală (Mică - ex: 96px)
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        self.local_crops_number = local_crops_number

    def __call__(self, image):
        crops = []
        # 2 Global Crops
        crops.append(self.global_transfo(image))
        crops.append(self.global_transfo(image))
        
        # N Local Crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
            
        return crops