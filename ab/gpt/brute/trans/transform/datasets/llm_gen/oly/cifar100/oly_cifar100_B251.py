import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomApply([transforms.Grayscale(p=1.0)], p=0.2), transforms.RandomPosterize(bits=4, p=0.3), transforms.Pad(padding=4, fill=(149, 238, 150), padding_mode='symmetric'), transforms.RandomCrop(size=32), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-5.0, 5.0, -5.0, 5.0)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])