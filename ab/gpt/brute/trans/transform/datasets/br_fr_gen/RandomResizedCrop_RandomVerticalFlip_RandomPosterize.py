import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.95), ratio=(1.18, 2.6)),
    transforms.RandomVerticalFlip(p=0.12),
    transforms.RandomPosterize(bits=8, p=0.25),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
