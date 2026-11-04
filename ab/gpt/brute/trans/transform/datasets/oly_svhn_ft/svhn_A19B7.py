import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=24), transforms.RandomAdjustSharpness(sharpness_factor=1.83, p=0.85), transforms.RandomVerticalFlip(p=0.86), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])