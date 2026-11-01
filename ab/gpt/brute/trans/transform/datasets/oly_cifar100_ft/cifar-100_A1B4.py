import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.74, 0.9), ratio=(1.02, 2.32)), transforms.RandomPosterize(bits=6, p=0.74), transforms.RandomGrayscale(p=0.42), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])