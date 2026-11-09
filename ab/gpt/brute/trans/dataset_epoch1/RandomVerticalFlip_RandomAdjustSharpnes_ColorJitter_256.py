import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.63),
    transforms.RandomAdjustSharpness(sharpness_factor=1.07, p=0.26),
    transforms.ColorJitter(brightness=1.12, contrast=0.89, saturation=0.91, hue=0.0),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
