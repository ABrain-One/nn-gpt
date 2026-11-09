import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=30),
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.96), ratio=(0.99, 1.5)),
    transforms.RandomGrayscale(p=0.86),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
