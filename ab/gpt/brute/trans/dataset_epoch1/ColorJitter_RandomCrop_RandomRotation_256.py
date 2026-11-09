import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.93, contrast=0.95, saturation=1.16, hue=0.06),
    transforms.RandomCrop(size=32),
    transforms.RandomRotation(degrees=22),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
