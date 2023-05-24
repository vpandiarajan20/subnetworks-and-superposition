import torch
from torchvision.datasets import ImageNet
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import os
from PIL import Image
import json


class SingleImage(Dataset):
    def __init__(self, img, label):
        self.img = img
        self.label = label
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.img, self.label


class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
                
    def __len__(self):
            return len(self.samples)
        
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]
        
        
class ImageNetReceptiveField(ImageNetKaggle):
    def __init__(self, root, split, transform=transforms.ToTensor(),
                background_c =(255,255,255), recep_field=None, img_size=224):
        
        super().__init__(root=root, split=split, transform=transform)
        
        self.root = root
        self.background_c = background_c
        self.img_size = img_size
    
        self.recep_field = recep_field
        
        if self.recep_field:
            self.recep_resize = transforms.Resize((int(recep_field[0][1]-recep_field[0][0]),int(recep_field[1][1]-recep_field[1][0]))) #size of receptive field 

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)

        if self.recep_field is not None:
            img_tensor = torch.zeros(3,self.img_size,self.img_size)
            shrunk_tensor = self.recep_resize(img)
            img_tensor[:,int(self.recep_field[0][0]):int(self.recep_field[0][1]),int(self.recep_field[1][0]):int(self.recep_field[1][1])] = shrunk_tensor
            img = img_tensor
     
        return img, label

class ImageNetClusters(ImageNetReceptiveField):
    def __init__(self, root, split, clusters, transform=transforms.ToTensor(),
                background_c =(255,255,255), recep_field=None, img_size=224):
        
        super().__init__(root, split, transform=transform,
                background_c =background_c, recep_field=recep_field, img_size=img_size)
        self.clusters = clusters
                
    def __len__(self):
        return super().__len__()
        
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        lab = self.clusters[idx]
        return img, lab


class ImageNet2(datasets.ImageFolder):
    def __init__(self, root, split, transform=transforms.ToTensor(),
                background_c =(255,255,255), recep_field=None, img_size=224):
        
        super().__init__(root=root, transform=transform)
        
        self.root = root
        self.background_c = background_c
        self.img_size = img_size
    
        self.recep_field = recep_field
        
        if self.recep_field:
            self.recep_resize = transforms.Resize((int(recep_field[0][1]-recep_field[0][0]),int(recep_field[1][1]-recep_field[1][0]))) #size of receptive field 

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)

        if self.recep_field is not None:
            img_tensor = torch.zeros(3,self.img_size,self.img_size)
            shrunk_tensor = self.recep_resize(img)
            img_tensor[:,int(self.recep_field[0][0]):int(self.recep_field[0][1]),int(self.recep_field[1][0]):int(self.recep_field[1][1])] = shrunk_tensor
            img = img_tensor
        return img, label