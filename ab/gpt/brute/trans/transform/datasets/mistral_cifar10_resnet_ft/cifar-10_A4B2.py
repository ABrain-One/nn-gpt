import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(size=28), transforms.RandomPerspective(distortion_scale=0.28, p=0.36), transforms.RandomEqualize(p=0.55), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])