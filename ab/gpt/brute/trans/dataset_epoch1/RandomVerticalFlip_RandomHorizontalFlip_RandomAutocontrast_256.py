import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.85),
    transforms.RandomHorizontalFlip(p=0.42),
    transforms.RandomAutocontrast(p=0.4),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
