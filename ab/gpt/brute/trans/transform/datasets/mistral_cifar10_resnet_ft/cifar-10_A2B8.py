import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.56), transforms.RandomAdjustSharpness(sharpness_factor=1.15, p=0.27), transforms.RandomCrop(size=31), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])