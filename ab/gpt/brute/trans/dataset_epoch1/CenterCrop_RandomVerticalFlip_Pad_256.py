import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=26),
    transforms.RandomVerticalFlip(p=0.14),
    transforms.Pad(padding=5, fill=(27, 95, 165), padding_mode='constant'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
