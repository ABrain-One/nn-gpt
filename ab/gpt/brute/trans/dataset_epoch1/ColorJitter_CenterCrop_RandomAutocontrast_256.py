import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.11, contrast=1.08, saturation=1.12, hue=0.06),
    transforms.CenterCrop(size=31),
    transforms.RandomAutocontrast(p=0.72),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
