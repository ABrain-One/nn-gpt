import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.RandomInvert(p=0.88),
    transforms.ColorJitter(brightness=1.11, contrast=0.83, saturation=0.83, hue=0.01),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
