import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(size=26), transforms.RandomAdjustSharpness(sharpness_factor=1.08, p=0.15), transforms.RandomInvert(p=0.15), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])