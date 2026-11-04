import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=30),
    transforms.RandomPosterize(bits=8, p=0.55),
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.82), ratio=(1.28, 1.53)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
