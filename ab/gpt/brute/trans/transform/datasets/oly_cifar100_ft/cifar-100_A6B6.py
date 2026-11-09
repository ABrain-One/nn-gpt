import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.77, 0.94), ratio=(0.8, 2.04)), transforms.RandomPosterize(bits=5, p=0.83), transforms.CenterCrop(size=30), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=norm[0], std=norm[1])])