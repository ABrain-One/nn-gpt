import torch
import torchvision.transforms as transforms

def transform(norm):
    pad_fill = (int(norm[0] * 255), int(norm[1] * 255), int(norm[2] * 255))
    return transforms.Compose([transforms.Pad(padding=4, fill=pad_fill, padding_mode='constant'), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.5, 2.0)), transforms.RandomPerspective(distortion_scale=0.2, p=0.5), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(*norm)])