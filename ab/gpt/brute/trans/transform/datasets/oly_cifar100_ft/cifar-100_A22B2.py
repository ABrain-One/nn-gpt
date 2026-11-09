import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.61), transforms.RandomResizedCrop(size=32, scale=(0.71, 0.99), ratio=(0.8, 1.62)), transforms.RandomSolarize(threshold=168, p=0.14), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.495, 0.46, 0.7), (0.281, 0.225, 0.289))])