import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(68, 165, 51), padding_mode='constant'), transforms.RandomResizedCrop(size=32, scale=(0.7, 1.0), ratio=(0.8, 1.3333333333333333)), transforms.RandomRotation(degrees=(-15, 15), expand=True), transforms.Resize(224), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.49139975, 0.48215855, 0.4465302], std=[0.24703233, 0.24348578, 0.2615876])])