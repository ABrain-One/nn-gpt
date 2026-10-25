import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.16),
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.99), ratio=(0.89, 2.29)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
