import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.61, 0.99), ratio=(1.01, 2.04)),
    transforms.RandomPosterize(bits=6, p=0.13),
    transforms.RandomHorizontalFlip(p=0.68),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
