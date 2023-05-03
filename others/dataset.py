import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import h5py
import math
from .augmentation import *
from typing import Tuple

class ColumbiaFaceDataset(Dataset):
    # Columbia Gaze Dataset (INPUT: PREPROCESSED CROPPED FACE IMAGE)
    def __init__(self, path: str, 
                 person_id: Tuple[int, ...],
                 augmentation=1, outSize=(224,224), 
                 gray=True):
        self.path = path
        self.gray = gray
        self.augmentation = augmentation
        self.person_id = person_id
        self.C = 1 if gray else 3
        self.outSize = outSize
        self.preprocess = transforms.Normalize([0.5] * self.C, [0.5] * self.C)
        self._augmentation()
        self.load_data()
        
    def _augmentation(self):
        if (self.augmentation):
            crop = RandomCrop(self.outSize, self.C)
            flip = RandomHorizontalFlip(self.C)
            blur = GaussianBlur(self.C)
            sharp = RandomAdjustSharpness(self.C)
            mixup1 = SelfMixup(self.outSize, self.C)
            mixup2 = SelfMixup(self.outSize, self.C)
            compose1 = transforms.Compose([RandomAdjustSharpness(self.C), RandomHorizontalFlip(self.C)])
            compose2 = transforms.Compose([GaussianBlur(self.C), RandomHorizontalFlip(self.C)])
            self.transform_list_aug = [blur, sharp, flip, crop, mixup1, mixup2, compose1, compose2]
            
    def __len__(self) -> int:
        return (len(self.images))
    
    def load_data(self):
        self.files_path = []
        for each in os.listdir(self.path):
            if each[:2] == '00' and int(each[2:4]) in self.person_id:
                self.files_path.extend([f.path for f in os.scandir(self.path+'/'+each+'/') 
                                        if f.is_file() and f.path[-8:]=='_new.jpg'])
                
        self.images, self.labels = [], []
        for person in self.files_path:
            with Image.open(person) as image:
                if self.gray: image = image.convert("L")
                image = transforms.ToTensor()(image)
                image_processed = self.preprocess(image)
                yaw, pitch = self.read_angles(person)
                
                self.images.append(image_processed)
                self.labels.append([yaw, pitch])
        
    def read_angles(self, name):
        filter = name.split("/")[-1].split("_")[3:]
        yaw = float(filter[0][:-1])
        pitch = float(filter[1].replace("H", ''))
        return yaw, pitch
    
    def __getitem__(
            self, 
            index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._augmentation()
        if self.augmentation:
            idx = np.random.randint(len(self.transform_list_aug) + 6)
            if idx >= len(self.transform_list_aug):
                return self.images[index], self.labels[index]
            else :
                image, labels = self.images[index], self.labels[index]
                method = self.transform_list_aug[idx]
                image, vert, hor = method((image, labels[0], labels[1]))
                return image, [vert, hor]
        return self.images[index], self.labels[index]
    
class MpiigazeEyeDataset(Dataset):
    def __init__(self, path: str, 
                 person_id: Tuple[int, ...], 
                 augmentation=True, outSize=(36,60)):
        self.path = path
        self.images, self.labels = [], []
        self.augmentation = augmentation
        self.person_id = person_id
        self.C = 1
        preprocess = transforms.ToTensor()
        for idx in person_id:
            person_id_str = str(idx)
            if len(person_id_str) == 1:
                person_id_str = 'p0' + person_id_str
            else:
                person_id_str = 'p' + person_id_str
            with h5py.File(path, 'r') as f:
                images = f.get(f'{person_id_str}/image')[()]
                labels = f.get(f'{person_id_str}/gaze')[()] * 180 / math.pi 
                labels[1] = -labels[1]
            preprocess = transforms.Compose([transforms.ToTensor(), 
                                             transforms.Resize(size=outSize)])
            for i in range(3000):
                image_matrix = preprocess(images[i])
                self.images.append(image_matrix)
                self.labels.append(labels[i] * 180/math.pi)
                if (augmentation):
                    for i, trsfrm in enumerate(self.transform_list_aug):
                        image_aug, yaw_aug, pitch_aug  = trsfrm((image_matrix, labels[i][0], labels[i][1]))
                        self.images.append(image_aug)
                        self.labels.append([yaw_aug, pitch_aug])
    
    def _augmentation(self):
        if (self.augmentation):
            crop = RandomCrop(self.outSize, self.C)
            flip = RandomHorizontalFlip(self.C)
            blur = GaussianBlur(self.C)
            sharp = RandomAdjustSharpness(self.C)
            mixup1 = SelfMixup(self.outSize, self.C)
            mixup2 = SelfMixup(self.outSize, self.C)
            compose1 = transforms.Compose([RandomAdjustSharpness(self.C), RandomHorizontalFlip(self.C)])
            compose2 = transforms.Compose([RandomAdjustSharpness(self.C), RandomCrop(self.outSize, self.C)])
            compose3 = transforms.Compose([GaussianBlur(self.C), RandomCrop(self.outSize, self.C)])
            compose4 = transforms.Compose([GaussianBlur(self.C), RandomHorizontalFlip(self.C)])
            self.trsfrm = [crop, flip, blur, sharp, 
                                       mixup1, mixup2, compose1, 
                                       compose2, compose3, compose4]

    def __getitem__(
            self,
            index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._augmentation()
        
        return self.images[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.images)
    
class MpiigazeFaceDataset(Dataset):
    def __init__(self, path: str, 
                 person_id: Tuple[int, ...], gray=True,
                 augmentation=1, outSize=(224,224)):
        self.path = path
        self.images, self.labels = [], []
        self.C = 1 if gray else 3
        self.augmentation = augmentation
        self.person_id = person_id
        self.outSize = outSize
        self.gray = gray
        self._augmentation()
        self._load_data()
                        
    def _load_data(self):
        for idx in self.person_id:
            person_id_str = str(idx)
            person_id_str = 'p0' + person_id_str if len(person_id_str) == 1 else 'p' + person_id_str
            for index in range(3000):
                with h5py.File(self.path, 'r') as f:
                    images = f.get(f'{person_id_str}/image/{index:04}')[()]
                    labels = f.get(f'{person_id_str}/gaze/{index:04}')[()] * 180/ math.pi
                labels[1] = -labels[1]
                
                preprocess = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Resize(size=self.outSize)])
                if self.gray:
                    preprocess = transforms.Compose([transforms.ToTensor(), 
                                                    transforms.Grayscale(),
                                                    transforms.Resize(size=self.outSize)])
                yaw, pitch = labels
                image_matrix = preprocess(images)
                self.images.append(image_matrix)
                self.labels.append([yaw, pitch])
                        
    def _augmentation(self):
        if (self.augmentation):
            crop = RandomCrop(self.outSize, self.C)
            flip = RandomHorizontalFlip(self.C)
            blur = GaussianBlur(self.C)
            sharp = RandomAdjustSharpness(self.C)
            mixup1 = SelfMixup(self.outSize, self.C)
            mixup2 = SelfMixup(self.outSize, self.C)
            compose1 = transforms.Compose([RandomAdjustSharpness(self.C), RandomHorizontalFlip(self.C)])
            compose2 = transforms.Compose([RandomAdjustSharpness(self.C), RandomCrop(self.outSize, self.C)])
            compose3 = transforms.Compose([GaussianBlur(self.C), RandomCrop(self.outSize, self.C)])
            compose4 = transforms.Compose([GaussianBlur(self.C), RandomHorizontalFlip(self.C)])
            self.trsfrm = [crop, flip, mixup1, mixup2, compose1, blur, sharp, compose4, compose3, compose2]
        else:
            self.trsfrm = []

    def __getitem__(
            self,
            index: int):
        self._augmentation()
        randIdx = np.random.randint(len(self.trsfrm)+9)
        if self.augmentation and randIdx < len(self.trsfrm):
            image, yaw, pitch = self.trsfrm[randIdx]((self.images[index], 
                                                     self.labels[index][0], 
                                                     self.labels[index][1]))
            labels = [yaw, pitch]
            return image, labels
        else: 
            return self.images[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.images)