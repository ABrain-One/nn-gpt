import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.8),
    transforms.ColorJitter(brightness=1.03, contrast=0.82, saturation=1.09, hue=0.07),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
