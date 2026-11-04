import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.5), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-5, 5), p=0.7), transforms.RandomAdjustSharpness(1.4, p=0.8), transforms.RandomApply([transforms.RandomSolarize(220, p=1.0)], p=0.3), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])