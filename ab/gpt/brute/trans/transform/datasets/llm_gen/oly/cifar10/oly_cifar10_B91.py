import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(0.6, 1.5)), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.RandomGrayscale(p=0.2), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2), transforms.RandomAdjustSharpness(1.8, p=0.2), transforms.RandomPerspective(distortion_scale=0.05, p=0.5), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])