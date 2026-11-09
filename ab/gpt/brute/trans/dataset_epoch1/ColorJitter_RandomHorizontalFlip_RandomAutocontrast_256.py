import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.11, contrast=1.16, saturation=1.07, hue=0.07),
    transforms.RandomHorizontalFlip(p=0.57),
    transforms.RandomAutocontrast(p=0.79),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
