import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=32),
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.98), ratio=(1.18, 1.83)),
    transforms.RandomGrayscale(p=0.71),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
