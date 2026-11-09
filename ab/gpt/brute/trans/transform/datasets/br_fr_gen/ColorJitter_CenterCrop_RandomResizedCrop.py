import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.88, contrast=0.99, saturation=1.0, hue=0.03),
    transforms.CenterCrop(size=26),
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.98), ratio=(0.88, 1.96)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
