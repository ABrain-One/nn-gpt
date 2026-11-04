import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.2), ratio=(1.0, 1.0)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), transforms.Pad(padding=2, fill=(78, 112, 144), padding_mode='symmetric'), transforms.RandomPerspective(distortion_scale=0.25, p=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])