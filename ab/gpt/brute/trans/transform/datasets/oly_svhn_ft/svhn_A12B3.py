import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.71, 0.9), ratio=(1.19, 1.44)), transforms.RandomGrayscale(p=0.43), transforms.RandomPosterize(bits=7, p=0.11), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])