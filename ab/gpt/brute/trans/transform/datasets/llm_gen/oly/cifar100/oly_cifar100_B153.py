import torch
import torchvision.transforms as transforms
from PIL import Image

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(210, 68, 250), padding_mode='symmetric'), transforms.RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5), resample=Image.BICUBIC, fill=(255, 255, 255)), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=1.5)], p=0.2), transforms.RandomErasing(scale=(0.04, 0.1), ratio=(0.8, 1.2), p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(**norm)])