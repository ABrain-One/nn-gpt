import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=23),
    transforms.RandomResizedCrop(size=32, scale=(0.65, 0.83), ratio=(0.95, 2.71)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
