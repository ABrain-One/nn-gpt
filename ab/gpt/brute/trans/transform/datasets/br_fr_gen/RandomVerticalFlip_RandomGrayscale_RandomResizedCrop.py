import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.58),
    transforms.RandomGrayscale(p=0.88),
    transforms.RandomResizedCrop(size=32, scale=(0.5, 0.88), ratio=(0.95, 1.42)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
