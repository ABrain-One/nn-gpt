import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-2, 2)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomSolarize(threshold=136, p=0.45), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])