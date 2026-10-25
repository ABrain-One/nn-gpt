import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(49, 201, 205), padding_mode='symmetric'),
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.92), ratio=(1.26, 2.99)),
    transforms.RandomInvert(p=0.82),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])