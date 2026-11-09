import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(254, 159, 147), padding_mode='edge'),
    transforms.RandomResizedCrop(size=32, scale=(0.8, 0.94), ratio=(1.05, 1.34)),
    transforms.RandomHorizontalFlip(p=0.84),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])