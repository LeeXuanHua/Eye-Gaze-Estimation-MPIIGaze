import torch
from torchvision import transforms     
import numpy as np
import os
import pandas as pd
from PIL import Image
    
class RandomCrop():
    def __init__(self, outSize, C, scale = (0.8,0.95)):
        self.outSize = outSize
        self.C = C
        self.scale = scale
        assert(len(outSize) == 2), f"Expected dim == 2 not {len(outSize)}"
        
    def __call__(self, sample):
        image, yaw, pitch = sample[0], sample[1], sample[2]
        h,w = self.outSize
        image_aug = transforms.RandomResizedCrop((h,w), scale=self.scale, ratio=(0.9, 1.1))(image)
        image_aug = (image_aug - torch.min(image_aug)) / (torch.max(image_aug)-torch.min(image_aug))
        image_aug = transforms.Normalize([0.5] * self.C, [0.5] * self.C)(image_aug)
        return image_aug, yaw, pitch
    
class RandomHorizontalFlip():
    def __init__(self, C):
        self.C = C
    def __call__(self, sample):
        image, yaw, pitch = sample[0], sample[1], sample[2]
        image_aug = transforms.RandomHorizontalFlip(p=1.0)(image)
        image_aug = (image_aug - torch.min(image_aug)) / (torch.max(image_aug)-torch.min(image_aug))
        image_aug = transforms.Normalize([0.5] * self.C, [0.5] * self.C)(image_aug)
        return image_aug, yaw, -pitch
    
class GaussianBlur():
    def __init__(self, C, kernel_size=(5,5), sigma=(0.9, 1.3)):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.C = C
        assert len(kernel_size) == 2, "Invalid kernel dims"
        
    def __call__(self, sample):
        image, yaw, pitch = sample[0], sample[1], sample[2]
        image_aug = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)(image)
        image_aug = (image_aug - torch.min(image_aug)) / (torch.max(image_aug)-torch.min(image_aug))
        image_aug = transforms.Normalize([0.5] * self.C, [0.5] * self.C)(image_aug)
        return image_aug, yaw, pitch

class RandomAdjustSharpness():
    def __init__(self, C):
        self.C = C
    def __call__(self, sample):
        image, yaw, pitch = sample[0], sample[1], sample[2]
        randsharp = 0.8 + 4 * np.random.rand(1)
        image_aug = transforms.RandomAdjustSharpness(randsharp[0], p=1.0)(image)
        image_aug = (image_aug - torch.min(image_aug)) / (torch.max(image_aug)-torch.min(image_aug))
        image_aug = transforms.Normalize([0.5] * self.C, [0.5] * self.C)(image_aug)
        return image_aug, yaw, pitch

class SelfMixup():
    def __init__(self, outSize, C):
        self.C = C
        self.outSize = outSize
    def __call__(self, sample):
        image_aug, yaw, pitch = RandomAdjustSharpness(self.C)(sample)
        image_aug_2, _, _ = RandomCrop(self.outSize, self.C, (0.95,0.98))(sample)
        image_aug = (image_aug_2 + image_aug)/2
        return image_aug, yaw, pitch

class SelfMixup_2():
    def __init__(self, outSize, C):
        self.C = C
        self.outSize = outSize
    def __call__(self, sample):
        image_aug, yaw, pitch = GaussianBlur(self.C)(sample)
        image_aug_2, _, _ = RandomCrop(self.outSize, self.C, (0.95,0.98))(sample)
        image_aug = (image_aug_2 + image_aug)/2
        return image_aug, yaw, pitch