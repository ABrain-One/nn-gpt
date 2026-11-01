import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.Pad(3, fill=(198, 154, 193), padding_mode='reflect'), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[0.7725, 0.596, 0.753], std=[0.259, 0.258, 0.257])])