import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.97), ratio=(0.96, 1.4)),
    transforms.RandomHorizontalFlip(p=0.22),
    transforms.Pad(padding=3, fill=(198, 154, 193), padding_mode='reflect'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
