import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.86, contrast=1.02, saturation=0.84, hue=0.03),
    transforms.RandomCrop(size=31),
    transforms.RandomAutocontrast(p=0.9),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
