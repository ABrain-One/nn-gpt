import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(243, 9, 131), padding_mode='constant'),
    transforms.RandomAffine(degrees=12, translate=(0.04, 0.17), scale=(0.86, 1.93), shear=(0.34, 6.73)),
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.93), ratio=(1.04, 2.8)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
