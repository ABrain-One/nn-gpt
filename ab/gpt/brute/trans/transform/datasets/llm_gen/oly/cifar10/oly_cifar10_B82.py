import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(4), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201]), transforms.Resize(224)])