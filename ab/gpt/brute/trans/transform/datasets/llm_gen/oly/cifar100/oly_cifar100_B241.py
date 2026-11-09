import torch
import torchvision.transforms as T
import random

class ApplyGaussianBlur:

    def __call__(self, img):
        if random.random() < 0.5:
            return T.functional.gaussian_blur(img, kernel_size=3, sigma=(0.5, 1.5))
        return img

def transform(norm):
    return T.Compose([T.RandomResizedCrop(32, scale=(0.5, 0.8), ratio=(1.2, 1.9)), T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), T.RandomRotation(degrees=(-5, 5)), T.RandomHorizontalFlip(p=0.5), ApplyGaussianBlur(), T.ToTensor(), T.Normalize(*norm)])