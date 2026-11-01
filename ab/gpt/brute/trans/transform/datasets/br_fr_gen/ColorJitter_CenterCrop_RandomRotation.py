import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.05, contrast=1.05, saturation=1.04, hue=0.01),
    transforms.CenterCrop(size=26),
    transforms.RandomRotation(degrees=1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
