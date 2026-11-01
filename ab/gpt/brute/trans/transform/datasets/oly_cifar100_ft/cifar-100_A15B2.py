import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.63, 0.94), ratio=(0.94, 2.4)), transforms.RandomPosterize(bits=7, p=0.53), transforms.RandomPerspective(distortion_scale=0.14, p=0.72), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])