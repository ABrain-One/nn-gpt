import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPosterize(bits=5, p=0.85),
    transforms.RandomResizedCrop(size=32, scale=(0.78, 1.0), ratio=(1.01, 1.44)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
