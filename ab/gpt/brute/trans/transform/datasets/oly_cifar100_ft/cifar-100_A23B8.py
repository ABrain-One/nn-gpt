import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=31), transforms.RandomInvert(p=0.52), transforms.RandomPosterize(bits=7, p=0.66), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])