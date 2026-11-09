import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.8), ratio=(0.8, 2.41)),
    transforms.RandomEqualize(p=0.23),
    transforms.ColorJitter(brightness=1.05, contrast=1.15, saturation=1.06, hue=0.02),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
