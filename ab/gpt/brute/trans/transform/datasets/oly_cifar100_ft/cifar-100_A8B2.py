import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.67), transforms.RandomResizedCrop(size=32, scale=(0.55, 1.05), ratio=(0.99, 2.22)), transforms.RandomPosterize(bits=4, p=0.63), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.49, 0.19, 0.81], std=[0.28, 0.46, 0.22])])