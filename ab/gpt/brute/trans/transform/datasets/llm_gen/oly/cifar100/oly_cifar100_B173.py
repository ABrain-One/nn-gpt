import torch
import torchvision.transforms as transforms

def transform():
    return transforms.Compose([transforms.Resize(72), transforms.RandomCrop(64, pad_if_needed=True, fill=(222, 206, 177), padding_mode='edge'), transforms.RandomHorizontalFlip(p=0.7), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])