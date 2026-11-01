import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=27),
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.91), ratio=(1.2, 2.53)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
