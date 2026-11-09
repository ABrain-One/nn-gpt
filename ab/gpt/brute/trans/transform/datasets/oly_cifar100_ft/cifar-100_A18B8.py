import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.39), transforms.RandomAdjustSharpness(sharpness_factor=0.51, p=0.35), transforms.RandomPosterize(bits=5, p=0.16), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.481, 0.461, 0.7), (0.288, 0.226, 0.283))])