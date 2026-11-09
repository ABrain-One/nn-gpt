import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.5, 2.0)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.2)], p=0.5), transforms.RandomAffine(degrees=15, translate=(0.06, 0.17), scale=(1.07, 1.88), shear=(4.65, 7.35)), transforms.ToTensor(), transforms.Normalize(**norm)])