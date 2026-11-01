import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.82),
    transforms.RandomPosterize(bits=8, p=0.65),
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.95), ratio=(1.12, 2.19)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])