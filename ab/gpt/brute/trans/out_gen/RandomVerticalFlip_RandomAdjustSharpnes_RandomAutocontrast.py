import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.66),
    transforms.RandomAdjustSharpness(sharpness_factor=0.91, p=0.7),
    transforms.RandomAutocontrast(p=0.45),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
