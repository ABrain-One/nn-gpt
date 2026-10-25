import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.18),
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.84), ratio=(0.95, 2.92)),
    transforms.RandomGrayscale(p=0.82),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
