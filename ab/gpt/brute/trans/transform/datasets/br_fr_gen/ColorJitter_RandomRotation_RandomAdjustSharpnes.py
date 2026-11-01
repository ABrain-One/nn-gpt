import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.98, contrast=1.1, saturation=0.96, hue=0.09),
    transforms.RandomRotation(degrees=23),
    transforms.RandomAdjustSharpness(sharpness_factor=1.89, p=0.72),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
