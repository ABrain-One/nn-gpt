import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=27), transforms.Resize(size=(64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.464, 0.447, 0.427], std=[0.202, 0.1993, 0.1957])])