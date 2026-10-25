import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.86), transforms.RandomPosterize(bits=4, p=0.42), transforms.RandomResizedCrop(size=32, scale=(0.73, 0.83), ratio=(1.16, 2.16)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])