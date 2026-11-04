import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.85),
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.86), ratio=(0.87, 2.19)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
