import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.21), transforms.RandomPosterize(bits=7, p=0.68), transforms.RandomEqualize(p=0.65), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])