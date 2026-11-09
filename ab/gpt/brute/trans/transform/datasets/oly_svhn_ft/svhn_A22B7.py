import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomPosterize(bits=4, p=0.89), transforms.RandomGrayscale(p=0.48), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])