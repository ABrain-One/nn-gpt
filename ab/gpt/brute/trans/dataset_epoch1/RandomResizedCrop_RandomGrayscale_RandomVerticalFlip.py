import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.5, 0.86), ratio=(0.97, 1.53)),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomVerticalFlip(p=0.82),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
