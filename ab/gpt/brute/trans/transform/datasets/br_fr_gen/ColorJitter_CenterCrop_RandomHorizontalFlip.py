import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.97, contrast=1.05, saturation=1.2, hue=0.06),
    transforms.CenterCrop(size=26),
    transforms.RandomHorizontalFlip(p=0.43),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
