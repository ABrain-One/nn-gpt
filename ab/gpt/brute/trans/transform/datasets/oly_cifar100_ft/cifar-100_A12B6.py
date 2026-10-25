import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.61, 0.89), ratio=(1.26, 1.82)), transforms.RandomGrayscale(p=0.13), transforms.GaussianBlur(kernel_size=5, sigma=(0.84, 1.71)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])