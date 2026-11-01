import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomPosterize(bits=4, p=0.8), transforms.RandomGrayscale(p=0.68), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])