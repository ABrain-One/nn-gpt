import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.17),
    transforms.RandomResizedCrop(size=32, scale=(0.62, 0.84), ratio=(1.05, 1.76)),
    transforms.Pad(padding=5, fill=(184, 199, 44), padding_mode='edge'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
