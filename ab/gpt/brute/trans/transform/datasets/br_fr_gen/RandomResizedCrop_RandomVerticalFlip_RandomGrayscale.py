import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.89), ratio=(1.01, 2.1)),
    transforms.RandomVerticalFlip(p=0.54),
    transforms.RandomGrayscale(p=0.52),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
