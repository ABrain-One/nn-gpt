import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.02, contrast=0.97, saturation=0.84, hue=0.03),
    transforms.RandomHorizontalFlip(p=0.35),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
