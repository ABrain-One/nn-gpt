import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.RandomGrayscale(p=0.75),
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.92), ratio=(1.06, 1.72)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
