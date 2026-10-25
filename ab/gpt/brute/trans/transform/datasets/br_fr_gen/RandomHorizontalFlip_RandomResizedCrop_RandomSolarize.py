import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.67),
    transforms.RandomResizedCrop(size=32, scale=(0.61, 0.82), ratio=(0.87, 2.32)),
    transforms.RandomSolarize(threshold=200, p=0.24),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
