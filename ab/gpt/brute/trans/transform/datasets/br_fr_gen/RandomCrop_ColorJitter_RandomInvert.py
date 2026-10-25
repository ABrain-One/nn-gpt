import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=27),
    transforms.ColorJitter(brightness=1.0, contrast=1.04, saturation=0.83, hue=0.05),
    transforms.RandomInvert(p=0.67),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
