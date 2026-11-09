import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=1.7, p=0.88), transforms.RandomRotation(degrees=13), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])