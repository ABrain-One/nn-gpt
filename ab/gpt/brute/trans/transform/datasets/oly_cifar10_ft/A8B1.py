import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomVerticalFlip(p=0.75),
        transforms.RandomRotation(degrees=9),
        transforms.RandomAdjustSharpness(sharpness_factor=1.13, p=0.71),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])