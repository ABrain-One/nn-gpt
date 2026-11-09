import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.14, contrast=1.01, saturation=1.12, hue=0.0),
    transforms.CenterCrop(size=30),
    transforms.RandomInvert(p=0.22),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
