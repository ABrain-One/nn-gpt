import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=28),
    transforms.RandomResizedCrop(size=32, scale=(0.77, 0.83), ratio=(0.75, 1.88)),
    transforms.CenterCrop(size=32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
