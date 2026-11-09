import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.68, 0.93), ratio=(1.21, 2.14)),
    transforms.RandomGrayscale(p=0.86),
    transforms.RandomRotation(degrees=26),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
