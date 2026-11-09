import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.7),
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.91), ratio=(1.24, 2.09)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
