import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.5, 0.9), ratio=(0.8, 3.0)), transforms.RandomInvert(p=0.25), transforms.RandomGrayscale(p=0.5), transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.2), scale=(0.8, 1.2), shear=(-10, 10)), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.4325, 0.4465, 0.4131], std=[0.2884, 0.2741, 0.2934])])