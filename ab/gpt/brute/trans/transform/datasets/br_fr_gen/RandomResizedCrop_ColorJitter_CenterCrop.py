import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.61, 0.87), ratio=(1.08, 2.73)),
    transforms.ColorJitter(brightness=0.99, contrast=0.89, saturation=1.05, hue=0.0),
    transforms.CenterCrop(size=29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
