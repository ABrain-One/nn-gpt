import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.94), ratio=(0.79, 2.12)),
    transforms.RandomRotation(degrees=22),
    transforms.RandomPosterize(bits=4, p=0.22),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
