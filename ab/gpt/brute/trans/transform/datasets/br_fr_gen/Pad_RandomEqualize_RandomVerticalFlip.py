import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(177, 161, 145), padding_mode='constant'),
    transforms.RandomEqualize(p=0.48),
    transforms.RandomVerticalFlip(p=0.8),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
