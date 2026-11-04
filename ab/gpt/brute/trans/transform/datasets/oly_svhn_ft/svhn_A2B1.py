import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.78, 0.84), ratio=(0.89, 2.91)), transforms.RandomAutocontrast(p=0.48), transforms.RandomGrayscale(p=0.62), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])