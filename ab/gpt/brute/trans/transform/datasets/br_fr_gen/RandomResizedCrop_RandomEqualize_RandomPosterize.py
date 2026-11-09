import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.86), ratio=(1.3, 1.71)),
    transforms.RandomEqualize(p=0.87),
    transforms.RandomPosterize(bits=5, p=0.85),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
