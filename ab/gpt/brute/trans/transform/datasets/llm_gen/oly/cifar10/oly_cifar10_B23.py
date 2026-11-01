import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(34), transforms.Grayscale(p=0.1), transforms.Pad(padding=3, padding_mode='symmetric', fill=(24, 123, 108)), transforms.RandomRotation(5), transforms.RandomHorizontalFlip(), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])