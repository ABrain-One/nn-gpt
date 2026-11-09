import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(0, 188, 54), padding_mode='constant'),
    transforms.RandomInvert(p=0.78),
    transforms.RandomResizedCrop(size=32, scale=(0.62, 0.85), ratio=(1.23, 2.3)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
