import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.17, contrast=0.92, saturation=0.87, hue=0.06),
    transforms.RandomGrayscale(p=0.18),
    transforms.RandomResizedCrop(size=32, scale=(0.59, 0.9), ratio=(0.76, 2.81)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
