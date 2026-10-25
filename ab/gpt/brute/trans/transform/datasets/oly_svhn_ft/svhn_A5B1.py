import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.4), transforms.RandomGrayscale(p=0.38), transforms.RandomPosterize(bits=4, p=0.54), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])