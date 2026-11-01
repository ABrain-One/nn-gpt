import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.45), transforms.RandomPosterize(bits=5, p=0.34), transforms.RandomGrayscale(p=0.24), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])