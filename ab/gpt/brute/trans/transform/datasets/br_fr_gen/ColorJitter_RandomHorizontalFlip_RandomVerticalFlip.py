import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.99, contrast=1.0, saturation=0.81, hue=0.0),
    transforms.RandomHorizontalFlip(p=0.66),
    transforms.RandomVerticalFlip(p=0.14),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
