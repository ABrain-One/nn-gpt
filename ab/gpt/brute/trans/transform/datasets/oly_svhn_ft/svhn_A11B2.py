import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.79, 0.93), ratio=(1.09, 2.04)), transforms.RandomGrayscale(p=0.26), transforms.RandomInvert(p=0.33), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])