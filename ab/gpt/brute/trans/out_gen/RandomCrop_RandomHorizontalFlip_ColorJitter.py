import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=25),
    transforms.RandomHorizontalFlip(p=0.52),
    transforms.ColorJitter(brightness=1.15, contrast=0.9, saturation=1.0, hue=0.03),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
