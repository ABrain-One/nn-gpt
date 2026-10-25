import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(175, 50, 166), padding_mode='reflect'),
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.87), ratio=(0.75, 2.65)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
