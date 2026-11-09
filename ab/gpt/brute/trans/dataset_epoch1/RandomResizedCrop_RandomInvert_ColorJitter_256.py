import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.99), ratio=(1.08, 1.82)),
    transforms.RandomInvert(p=0.57),
    transforms.ColorJitter(brightness=1.06, contrast=0.83, saturation=1.13, hue=0.01),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
