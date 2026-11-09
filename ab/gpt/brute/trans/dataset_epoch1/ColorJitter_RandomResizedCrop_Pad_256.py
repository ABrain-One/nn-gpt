import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.82, contrast=0.99, saturation=0.88, hue=0.01),
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.84), ratio=(0.87, 1.64)),
    transforms.Pad(padding=4, fill=(90, 248, 104), padding_mode='constant'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
