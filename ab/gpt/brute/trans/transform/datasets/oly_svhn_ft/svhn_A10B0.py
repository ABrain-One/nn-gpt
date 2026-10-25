import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomPosterize(bits=4, p=0.78), transforms.RandomAdjustSharpness(sharpness_factor=1.9, p=0.14), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4943, 0.4811, 0.4804], std=[0.2001, 0.1979, 0.1925])])