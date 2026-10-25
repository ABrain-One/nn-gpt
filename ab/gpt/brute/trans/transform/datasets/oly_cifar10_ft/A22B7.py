import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.84),
        transforms.RandomCrop(size=29),
        transforms.RandomVerticalFlip(p=0.48),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])