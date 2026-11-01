import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random

class GaussianBlur(object):

    def __init__(self, kernel_size, sigma):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        if random.random() < 0.5:
            return F.gaussian_blur(img, self.kernel_size, self.sigma)
        return img

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, padding_mode='reflect'), transforms.RandomPerspective(distortion_scale=0.3, p=0.35), transforms.RandomResizedCrop(size=32, scale=(0.6, 1.0), ratio=(0.7, 1.2)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])