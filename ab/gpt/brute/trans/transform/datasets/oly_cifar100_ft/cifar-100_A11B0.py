import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=32), transforms.RandomSolarize(threshold=241, p=0.55), transforms.RandomVerticalFlip(p=0.22), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])