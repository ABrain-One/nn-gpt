import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.8, 0.87), ratio=(1.25, 2.48)),
    transforms.RandomGrayscale(p=0.78),
    transforms.RandomCrop(size=28),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
