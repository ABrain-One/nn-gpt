import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.7, 0.84), ratio=(1.32, 1.55)), transforms.RandomVerticalFlip(p=0.12), transforms.Pad(5, fill=tuple(norm[0]), padding_mode='reflect'), transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(p=0.11), transforms.ToTensor(), transforms.Normalize(*norm)])