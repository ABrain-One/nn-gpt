import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.RandomResizedCrop(size=32, scale=(0.65, 0.92), ratio=(0.84, 2.35)),
    transforms.ColorJitter(brightness=1.09, contrast=0.8, saturation=1.01, hue=0.08),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
