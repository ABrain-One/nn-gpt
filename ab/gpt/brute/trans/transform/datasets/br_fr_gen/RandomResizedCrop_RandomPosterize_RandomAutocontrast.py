import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.83), ratio=(1.23, 2.16)),
    transforms.RandomPosterize(bits=8, p=0.65),
    transforms.RandomAutocontrast(p=0.57),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
