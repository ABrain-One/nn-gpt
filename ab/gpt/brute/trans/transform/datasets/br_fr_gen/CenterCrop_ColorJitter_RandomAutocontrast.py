import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=29),
    transforms.ColorJitter(brightness=1.16, contrast=1.01, saturation=0.82, hue=0.02),
    transforms.RandomAutocontrast(p=0.11),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
