import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(94, 78, 222), padding_mode='reflect'),
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.8), ratio=(1.31, 2.12)),
    transforms.RandomPerspective(distortion_scale=0.28, p=0.68),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
