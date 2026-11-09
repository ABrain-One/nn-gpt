import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(240, 95, 30), padding_mode='constant'),
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.96), ratio=(1.21, 2.04)),
    transforms.RandomEqualize(p=0.19),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
