import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.87), ratio=(0.94, 2.61)),
    transforms.RandomRotation(degrees=28),
    transforms.RandomEqualize(p=0.16),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
