import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.95), ratio=(0.83, 2.7)),
    transforms.RandomInvert(p=0.4),
    transforms.RandomPosterize(bits=8, p=0.12),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
