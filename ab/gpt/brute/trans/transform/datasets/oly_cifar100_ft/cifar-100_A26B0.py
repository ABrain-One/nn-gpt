import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.54), transforms.RandomGrayscale(p=0.17), transforms.RandomResizedCrop(size=32, scale=(0.75, 0.87), ratio=(0.79, 1.37)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])