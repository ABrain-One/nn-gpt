import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(250, 191, 124), padding_mode='edge'),
    transforms.RandomRotation(degrees=8),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48], std=[0.91], max_val=255.0, inplace=True)
])