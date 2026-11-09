import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.23),
    transforms.RandomAutocontrast(p=0.33),
    transforms.RandomHorizontalFlip(p=0.18),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
