import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.89), ratio=(0.99, 2.95)),
    transforms.RandomCrop(size=25),
    transforms.Pad(padding=5, fill=(125, 213, 30), padding_mode='edge'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
