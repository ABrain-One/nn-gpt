import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.92), ratio=(1.28, 1.68)),
    transforms.RandomCrop(size=25),
    transforms.RandomEqualize(p=0.38),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
