import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomEqualize(p=0.23), transforms.RandomPosterize(bits=6, p=0.12), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])