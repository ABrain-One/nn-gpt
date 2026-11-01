import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomPosterize(5, p=0.47), transforms.RandomCrop(29), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])