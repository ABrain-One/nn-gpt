import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.51, 0.81), ratio=(0.93, 2.63)),
    transforms.RandomHorizontalFlip(p=0.11),
    transforms.RandomVerticalFlip(p=0.62),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
