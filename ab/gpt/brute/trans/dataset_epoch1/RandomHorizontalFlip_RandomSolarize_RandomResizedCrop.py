import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.34),
    transforms.RandomSolarize(threshold=14, p=0.45),
    transforms.RandomResizedCrop(size=32, scale=(0.68, 0.9), ratio=(0.94, 2.58)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
