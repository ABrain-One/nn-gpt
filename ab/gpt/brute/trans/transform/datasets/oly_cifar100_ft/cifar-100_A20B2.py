import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.71, 0.97), ratio=(1.03, 1.61)), transforms.RandomPosterize(bits=4, p=0.13), transforms.RandomPerspective(distortion_scale=0.18, p=0.57), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])