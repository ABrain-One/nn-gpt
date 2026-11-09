import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=24),
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.96), ratio=(1.26, 2.23)),
    transforms.RandomPosterize(bits=5, p=0.42),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
