from utils.helpers import *

paths = {
    "Finetunning": r"data\ARCADE\processed\dataset.json",
    "Pretraining": r"data\pretrain\pretrain.json"  #cadica | coronarydominance | syntax | xcad
}

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


data = load_data(paths["Finetunning"]) #train/val/test -> stenoza/syntax -> 1-n -> data/label
# split = "train"
# task = "stenoza"
# test = []
# for pacienti in data[split][task]:
#    test.append(data[split][task][pacienti])
# for i in range(1):
#     print(test[i])



class finetune_dataset(Dataset):
    def __init__(self, path=paths["Finetunning"], split="train", task='stenoza', transform=[]):
        '''
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
    
    


pretrain = load_data(paths["Pretraining"])
pacienti = {}

for splits in pretrain:
    pacienti[splits] = pretrain[splits]
    print(f"Split: {splits}, total: {len(pretrain[splits])}")

for pacient in pacienti["cadica"].values():
    print(pacient)
    break

#     print(f"Split: {splits}, total: {len(pretrain[splits])}")
# Split: cadica, total: 6594
# Split: coronarydominance, total: 160320
# Split: syntax, total: 2943
# Split: xcad, total: 1621    


class pretrain_dataset(Dataset):
    def __init__(self, path=paths["Pretraining"], split=None, transform=None):
        self.split = split
        self.transform = transform
        self.data = load_data(path)
        self.samples = []
        
        if split is None:
            for splits in self.data:
                pacienti = self.data[splits]
                self.samples.append(pacienti)
  
                    
    
# if __name__ == "__main__":
#     dataset = finetune_dataset(split="train", task="syntax")
#     #no transforms passed, so the images and labels will be returned as PIL images.
#     for image, label in dataset:
#         figure = plt.figure(figsize=(10, 5))
#         figure.add_subplot(1, 2, 1)
#         plt.imshow(image)
#         figure.add_subplot(1, 2, 2)
#         plt.imshow(label)
#         plt.show()
#         break
    

        