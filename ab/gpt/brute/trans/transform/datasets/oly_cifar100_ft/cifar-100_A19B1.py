import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=29), transforms.RandomPerspective(distortion_scale=0.23, p=0.53), transforms.RandomResizedCrop(size=32, scale=(0.71, 0.91), ratio=(1.01, 2.35)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])