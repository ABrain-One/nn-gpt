import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.7), transforms.RandomPosterize(bits=4, p=0.44), transforms.RandomCrop(size=28), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])