import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.6, 1.4), ratio=(0.8, 1.2)), transforms.RandomAutocontrast(p=0.8), transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5), transforms.Pad(padding=5, fill=(173, 254, 84), padding_mode='edge'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])