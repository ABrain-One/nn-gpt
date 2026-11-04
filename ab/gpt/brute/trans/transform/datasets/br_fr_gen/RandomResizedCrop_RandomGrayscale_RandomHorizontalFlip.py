import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.79, 0.92), ratio=(0.8, 1.73)),
    transforms.RandomGrayscale(p=0.54),
    transforms.RandomHorizontalFlip(p=0.48),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
