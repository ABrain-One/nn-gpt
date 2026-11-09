import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.62, 0.92), ratio=(0.93, 1.37)),
    transforms.RandomEqualize(p=0.19),
    transforms.RandomCrop(size=30),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
