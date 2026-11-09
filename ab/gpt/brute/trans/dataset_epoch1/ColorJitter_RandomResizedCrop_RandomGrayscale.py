import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.2, contrast=1.11, saturation=0.93, hue=0.05),
    transforms.RandomResizedCrop(size=32, scale=(0.52, 0.87), ratio=(1.02, 1.57)),
    transforms.RandomGrayscale(p=0.38),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
