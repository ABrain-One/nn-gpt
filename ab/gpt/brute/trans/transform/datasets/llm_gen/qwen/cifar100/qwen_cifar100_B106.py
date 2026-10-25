import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), transforms.RandomAffine(10), transforms.ToTensor(), transforms.Normalize(*norm)])