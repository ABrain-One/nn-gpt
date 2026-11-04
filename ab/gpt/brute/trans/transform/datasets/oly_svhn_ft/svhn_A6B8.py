import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.74, 0.88), ratio=(1.22, 2.36)), transforms.RandomGrayscale(p=0.48), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])