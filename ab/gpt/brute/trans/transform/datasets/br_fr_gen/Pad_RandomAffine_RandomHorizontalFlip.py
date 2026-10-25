import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(112, 122, 29), padding_mode='edge'),
    transforms.RandomAffine(degrees=20, translate=(0.07, 0.17), scale=(0.9, 1.38), shear=(2.23, 7.94)),
    transforms.RandomHorizontalFlip(p=0.16),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
