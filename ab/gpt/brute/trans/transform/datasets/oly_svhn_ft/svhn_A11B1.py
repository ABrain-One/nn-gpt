import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=29), transforms.RandomAdjustSharpness(sharpness_factor=1.31, p=0.35), transforms.RandomPosterize(bits=4, p=0.7), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])