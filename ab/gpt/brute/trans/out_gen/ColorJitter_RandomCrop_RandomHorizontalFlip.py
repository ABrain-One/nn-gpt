import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.99, contrast=1.06, saturation=0.96, hue=0.08),
    transforms.RandomCrop(size=32),
    transforms.RandomHorizontalFlip(p=0.53),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
