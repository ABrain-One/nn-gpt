import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(246, 205, 43), padding_mode='edge'),
    transforms.ColorJitter(brightness=0.81, contrast=0.83, saturation=1.2, hue=0.07),
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.93), ratio=(1.03, 2.35)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
