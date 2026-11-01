import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.73), transforms.RandomGrayscale(p=0.13), transforms.RandomPosterize(bits=7, p=0.13), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])