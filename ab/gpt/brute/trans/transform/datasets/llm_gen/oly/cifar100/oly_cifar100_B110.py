import torch
from torchvision import transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(237, 244, 221), padding_mode='constant'), transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5), resample=transforms.InterpolationMode.BICUBIC, fill=(237, 244, 221)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomEqualize(p=0.2), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])