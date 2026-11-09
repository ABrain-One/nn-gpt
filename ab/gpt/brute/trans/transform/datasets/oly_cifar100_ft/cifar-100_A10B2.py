import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(157, 48, 68), padding_mode='reflect'), transforms.RandomVerticalFlip(p=0.52), transforms.RandomResizedCrop(size=32, scale=(0.72, 0.83), ratio=(0.81, 2.0)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])