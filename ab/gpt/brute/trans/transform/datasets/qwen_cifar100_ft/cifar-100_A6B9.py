import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(213, 248, 171), padding_mode='symmetric'), transforms.RandomAdjustSharpness(sharpness_factor=1.47, p=0.21), transforms.RandomResizedCrop(size=32, scale=(0.64, 0.92), ratio=(0.88, 2.89)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])