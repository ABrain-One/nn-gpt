import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=12),
    transforms.Pad(padding=5, fill=(247, 76, 186), padding_mode='constant'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
