import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.87), ratio=(1.07, 1.97)),
    transforms.CenterCrop(size=27),
    transforms.Pad(padding=5, fill=(148, 149, 171), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
