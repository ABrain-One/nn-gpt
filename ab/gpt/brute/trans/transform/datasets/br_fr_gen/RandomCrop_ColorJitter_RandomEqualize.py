import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=24),
    transforms.ColorJitter(brightness=0.85, contrast=0.96, saturation=0.82, hue=0.07),
    transforms.RandomEqualize(p=0.54),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
