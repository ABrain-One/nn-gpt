import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(236, 47, 148), padding_mode='constant'),
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.93), ratio=(1.08, 2.57)),
    transforms.RandomInvert(p=0.4),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
