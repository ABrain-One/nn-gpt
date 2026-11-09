import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=25),
    transforms.Pad(padding=2, fill=(173, 128, 230), padding_mode='constant'),
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.87), ratio=(0.92, 2.43)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
