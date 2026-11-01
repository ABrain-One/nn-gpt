import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.61), transforms.RandomSolarize(threshold=146, p=0.15), transforms.RandomAutocontrast(p=0.24), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])