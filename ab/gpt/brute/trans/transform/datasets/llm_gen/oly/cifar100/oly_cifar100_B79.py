import torch
import torchvision.transforms as transforms

def transform():
    return transforms.Compose([transforms.Pad(padding=4, fill=(26, 133, 164), padding_mode='symmetric'), transforms.RandomResizedCrop(64, scale=(0.5, 1.0), ratio=(0.8, 2.0)), transforms.RandomAdjustSharpness(1.89, p=0.25), transforms.RandomHorizontalFlip(p=0.45), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])