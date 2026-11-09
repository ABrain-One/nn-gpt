import torch
import torchvision.transforms as transforms
from PIL import Image

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAffine(degrees=(-10, 10), translate_percent=(0.05, 0.05), scale=(0.9, 1.1), resample=Image.BICUBIC, fill=(227, 82, 213)), transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.5), transforms.RandomAutocontrast(p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])