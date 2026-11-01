import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPosterize(bits=7, p=0.47),
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.87), ratio=(1.15, 2.28)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
