import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(91, 234, 156), padding_mode='edge'),
    transforms.RandomResizedCrop(size=32, scale=(0.77, 0.86), ratio=(1.26, 2.83)),
    transforms.ColorJitter(brightness=1.09, contrast=0.92, saturation=1.1, hue=0.02),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
