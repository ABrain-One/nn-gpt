import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.13),
    transforms.Pad(padding=4, fill=(177, 150, 67), padding_mode='edge'),
    transforms.RandomHorizontalFlip(p=0.86),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
