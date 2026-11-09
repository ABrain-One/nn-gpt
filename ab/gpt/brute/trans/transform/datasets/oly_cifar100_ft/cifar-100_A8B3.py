import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=30), transforms.RandomRotation(degrees=10), transforms.RandomPosterize(bits=4, p=0.86), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])