import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.28),
    transforms.RandomResizedCrop(size=32, scale=(0.68, 0.91), ratio=(0.86, 1.61)),
    transforms.RandomGrayscale(p=0.71),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
