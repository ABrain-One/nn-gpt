import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(**norm)])