import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(212, 110, 63), padding_mode='reflect'), transforms.RandomResizedCrop(size=32, scale=(0.71, 0.92), ratio=(0.88, 2.43)), transforms.RandomAdjustSharpness(sharpness_factor=1.8, p=0.88), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])