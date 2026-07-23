from utils.helpers import *
import torchvision.transforms.functional as TF

paths = {
    "Finetunning": r"data\ARCADE\processed\dataset.json",
    "Pretraining": r"data\pretrain\pretrain.json"  #cadica | coronarydominance | syntax | xcad
}

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


class finetune_dataset(Dataset):
    def __init__(self, path=paths["Finetunning"], split="train", task='stenoza', transform=[]):
        '''
        RETURNS: No tranforms - PIL images, with tranforms: tensors - image, label. 
        Special Args:
            tranforms: A dictionary of transforms for the data and label. The keys should be "data" and "label" respectively.
            See implementation of data/data.py in __getitem__ for an example of how to pass the transforms.
            If no transforms are passed, the images and labels will be returned as PIL images.
        '''
        
        self.split = split
        self.task = task
        self.transform = transform
        self.data = load_data(path)
        self.samples = []
        
        for pacienti in self.data[split][task]:
            pacienti_data = self.data[split][task][pacienti]
            self.samples.append(
                {
                    'data': pacienti_data['data'],
                    'label': pacienti_data['label']
                }
            )
            
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['data'])
        label = Image.open(sample['label'])
        
        '''
        Tranforms require to be passed as an dictionary with keys 
        "data" and "label" for the respective transformations.
        e.g
        
        image_tranforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.ToTensor()
        ])
        
        label_tranforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        self.transform = {
            "data": image_tranforms,
            "label": label_tranforms
        }
        
        '''
        
        if self.transform:
            image = self.transform["data"](image)
            label = self.transform["label"](label) 
        
        return image, label
    

class pretrain_dataset(Dataset):
    def __init__(self, path=paths["Pretraining"], split=None, transform=None):
        self.split = split
        self.transform = transform
        self.data = load_data(path)
        self.samples = []
        
        if split is None:
            for splits in self.data:
                for pacient_id, image_path in self.data[splits].items():
                    self.samples.append(image_path)
  
        if split is not None:
            for patient_id, image_path in self.data[split].items():
                self.samples.append(image_path)
                
        # Derive prior path: replace last "dataset" or "data" dir with "priors", .png -> .npy
        priors = []
        for s in self.samples:
            if "/dataset/" in s or "\\dataset\\" in s:
                p = re.sub(r"([/\\])dataset([/\\])", r"\1priors\2", s)
            else:
                p = re.sub(r"([/\\])data([/\\])", r"\1priors\2", s)
            priors.append(p.rsplit(".", 1)[0] + ".npy")
        self.priors = priors
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert("L")
        prior = np.load(self.priors[idx]).astype(np.float32)   # [14, 14], values in [0, 1]

        # Synchronized horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            prior = np.fliplr(prior).copy()

        # Image: resize -> tensor -> normalize
        image = TF.resize(image, [224, 224])
        image = TF.to_tensor(image)                            # [1, 224, 224]
        image = TF.normalize(image, mean=[0.5], std=[0.5])

        # Prior: numpy -> tensor, no normalization (already [0, 1])
        prior = torch.from_numpy(prior)                        # [14, 14]

        return image, prior
    
    

if __name__ == "__main__":
    # set_seed()
    # pretrain = pretrain_dataset()
    # for image, priors in pretrain:
    #     plt.figure(figsize=(10, 5))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(image, cmap='gray')

    #     plt.subplot(1, 2, 2)
    #     plt.imshow(priors, cmap='gray')
    #     plt.show()
    #     break
    
    priors = np.load("data/pretrain/priors/cadica/.npy")

    print(f"Shape: {priors.shape}")
    print(f"Dtype: {priors.dtype}")
    print(f"Min: {priors.min()}")
    print(f"Max: {priors.max()}")
    print(f"Mean: {priors.mean()}")
    
                    

        
# pretrain = load_data(paths["Pretraining"])
# pacienti = {}

# for splits in pretrain:
#     pacienti[splits] = pretrain[splits]
#     print(f"Split: {splits}, total: {len(pretrain[splits])}")

# for pacient in pacienti["cadica"].values():
#     print(pacient)
#     break
#data/pretrain/dataset/cadica/1.png

# for splits in pretrain:
#     pacienti[splits] = pretrain[splits]

# for i in range(1):    
#     print(pacienti["syntax"])
#     break
#'2943': 'data/pretrain/dataset/syntax/2943.png


#     print(f"Split: {splits}, total: {len(pretrain[splits])}")
# Split: cadica, total: 6594
# Split: coronarydominance, total: 160320
# Split: syntax, total: 2943
# Split: xcad, total: 1621    



# data = load_data(paths["Finetunning"]) #train/val/test -> stenoza/syntax -> 1-n -> data/label
# # split = "train"
# # task = "stenoza"
# # test = []
# # for pacienti in data[split][task]:
# #    test.append(data[split][task][pacienti])
# # for i in range(1):
# #     print(test[i])

