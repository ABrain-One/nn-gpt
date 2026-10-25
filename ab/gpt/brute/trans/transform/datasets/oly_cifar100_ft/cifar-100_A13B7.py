import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.18), transforms.RandomPosterize(bits=7, p=0.6), transforms.RandomCrop(size=29), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.481, 0.462, 0.657], std=[0.251, 0.228, 0.421])])