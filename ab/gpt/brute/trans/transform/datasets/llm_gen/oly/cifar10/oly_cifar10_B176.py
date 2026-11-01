import torch
import torchvision.transforms as transforms
from PIL import Image

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(128, 128, 128), padding_mode='constant'), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5), resample=Image.BICUBIC, fill=128), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.RandomAutocontrast(p=0.5), transforms.RandomSolarize(threshold=30, p=0.5)], p=1.0), transforms.RandomCrop(24), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])