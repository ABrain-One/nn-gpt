import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=23),
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.98), ratio=(1.02, 1.47)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
