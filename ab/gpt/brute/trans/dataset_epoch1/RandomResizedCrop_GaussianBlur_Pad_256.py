import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.77, 0.82), ratio=(0.87, 2.13)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.84, 1.23)),
    transforms.Pad(padding=3, fill=(103, 36, 31), padding_mode='constant'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
