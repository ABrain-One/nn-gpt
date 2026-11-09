import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=31), transforms.RandomCrop(size=28), transforms.RandomSolarize(threshold=242, p=0.15), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])