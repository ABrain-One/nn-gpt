import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.75, 1.333)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.2), transforms.RandomApply([transforms.Solarize(threshold=190, p=1.0)], p=0.35), transforms.RandomAutocontrast(p=0.35), transforms.ToTensor(), transforms.Normalize(mean=[0.49139975, 0.48215856, 0.4465302], std=[0.2023, 0.1994, 0.201])])