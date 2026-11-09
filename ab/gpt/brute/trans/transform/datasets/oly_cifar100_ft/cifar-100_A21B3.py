import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=30), transforms.RandomGrayscale(p=0.13), transforms.RandomPosterize(bits=7, p=0.23), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])