import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.Pad(padding=1, fill=(148, 8, 165), padding_mode='constant'),
        transforms.RandomCrop(size=31),
        transforms.RandomResizedCrop(size=32, scale=(0.69, 0.9), ratio=(1.08, 2.92)),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])