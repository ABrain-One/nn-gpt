import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.86),
    transforms.ColorJitter(brightness=1.17, contrast=1.09, saturation=0.8, hue=0.07),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
