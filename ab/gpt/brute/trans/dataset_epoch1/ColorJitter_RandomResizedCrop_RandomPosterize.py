import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.83, contrast=1.19, saturation=0.85, hue=0.03),
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.96), ratio=(0.82, 2.68)),
    transforms.RandomPosterize(bits=5, p=0.54),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
