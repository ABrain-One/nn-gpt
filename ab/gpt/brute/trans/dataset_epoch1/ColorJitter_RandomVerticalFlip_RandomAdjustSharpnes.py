import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.18, contrast=0.88, saturation=0.85, hue=0.07),
    transforms.RandomVerticalFlip(p=0.48),
    transforms.RandomAdjustSharpness(sharpness_factor=1.77, p=0.56),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
