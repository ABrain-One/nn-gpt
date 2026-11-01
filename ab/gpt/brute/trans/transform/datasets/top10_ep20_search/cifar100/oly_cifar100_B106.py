import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.5), transforms.RandomApply([transforms.RandomPosterize(bits=4)], p=0.5), transforms.RandomApply([transforms.RandomAutocontrast()], p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])