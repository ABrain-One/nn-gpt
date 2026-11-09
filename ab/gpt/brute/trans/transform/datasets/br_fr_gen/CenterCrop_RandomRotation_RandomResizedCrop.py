import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=25),
    transforms.RandomRotation(degrees=21),
    transforms.RandomResizedCrop(size=32, scale=(0.52, 0.97), ratio=(1.12, 2.39)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
