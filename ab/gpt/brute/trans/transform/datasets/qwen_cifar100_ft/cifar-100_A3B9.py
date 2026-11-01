import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=0.98, p=0.65), transforms.RandomResizedCrop(size=32, scale=(0.73, 0.98), ratio=(0.82, 2.94)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])