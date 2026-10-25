import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.54, 1.0), ratio=(0.77, 2.78)), transforms.RandomHorizontalFlip(), transforms.RandomEqualize(p=0.27), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.GaussianBlur(3, sigma=1.0), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(**norm)])