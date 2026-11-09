import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(9, 200, 159), padding_mode='reflect'),
    transforms.RandomPerspective(distortion_scale=0.23, p=0.35),
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.92), ratio=(0.8, 2.11)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
