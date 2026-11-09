import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.99), ratio=(1.2, 1.58)),
    transforms.RandomGrayscale(p=0.2),
    transforms.CenterCrop(size=28),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
