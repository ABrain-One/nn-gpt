import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(31), transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(0.8, 1.2)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomGrayscale(p=0.47), transforms.RandomPerspective(distortion_scale=0.16, p=0.75), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(**norm)])