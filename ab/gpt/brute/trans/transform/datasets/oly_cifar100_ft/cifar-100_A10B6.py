import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=31), transforms.RandomResizedCrop(size=32, scale=(0.62, 0.86), ratio=(1.02, 1.44)), transforms.RandomAdjustSharpness(sharpness_factor=1.05, p=0.4), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])