import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.2)), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5), transforms.ColorJitter(0.3, 0.3, 0.3, 0.1), transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(5, resample=False, expansion=True), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(**norm)])