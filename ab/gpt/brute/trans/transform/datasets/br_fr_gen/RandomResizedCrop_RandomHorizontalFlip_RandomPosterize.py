import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.98), ratio=(0.83, 1.95)),
    transforms.RandomHorizontalFlip(p=0.67),
    transforms.RandomPosterize(bits=6, p=0.88),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
