import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=0.9, p=0.62),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
