import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.38),
    transforms.RandomPosterize(bits=6, p=0.85),
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.94), ratio=(0.78, 2.9)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
