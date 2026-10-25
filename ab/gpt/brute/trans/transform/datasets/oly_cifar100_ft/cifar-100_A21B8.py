import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.74, 0.95), ratio=(0.8, 2.0)), transforms.RandomGrayscale(p=0.15), transforms.RandomPosterize(bits=7, p=0.12), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.495, 0.461, 0.7), (0.288, 0.226, 0.228))])