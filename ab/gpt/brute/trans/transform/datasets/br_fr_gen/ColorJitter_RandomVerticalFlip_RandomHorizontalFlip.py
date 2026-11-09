import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.08, contrast=0.91, saturation=0.82, hue=0.04),
    transforms.RandomVerticalFlip(p=0.52),
    transforms.RandomHorizontalFlip(p=0.76),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
