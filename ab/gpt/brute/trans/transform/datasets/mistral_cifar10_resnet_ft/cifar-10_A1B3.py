import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.82), transforms.RandomPosterize(bits=7, p=0.16), transforms.RandomPerspective(distortion_scale=0.18, p=0.14), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])