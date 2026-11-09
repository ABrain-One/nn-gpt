import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.75),
    transforms.CenterCrop(size=24),
    transforms.ColorJitter(brightness=1.09, contrast=0.99, saturation=1.0, hue=0.06),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
