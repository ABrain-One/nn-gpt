import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.74),
    transforms.RandomAdjustSharpness(sharpness_factor=0.59, p=0.84),
    transforms.RandomRotation(degrees=3),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])