import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.89, contrast=1.14, saturation=1.04, hue=0.07),
    transforms.RandomCrop(size=31),
    transforms.RandomVerticalFlip(p=0.31),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
