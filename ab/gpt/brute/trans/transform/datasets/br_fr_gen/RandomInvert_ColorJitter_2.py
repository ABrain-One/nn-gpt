import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.52),
    transforms.ColorJitter(brightness=1.03, contrast=1.2, saturation=1.13, hue=0.08),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
