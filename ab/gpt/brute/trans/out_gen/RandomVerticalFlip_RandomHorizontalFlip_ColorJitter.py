import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.8),
    transforms.RandomHorizontalFlip(p=0.45),
    transforms.ColorJitter(brightness=1.0, contrast=1.19, saturation=1.04, hue=0.07),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
