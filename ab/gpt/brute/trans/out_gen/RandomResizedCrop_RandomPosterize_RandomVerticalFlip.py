import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.7, 1.0), ratio=(0.8, 1.65)),
    transforms.RandomPosterize(bits=5, p=0.17),
    transforms.RandomVerticalFlip(p=0.12),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
