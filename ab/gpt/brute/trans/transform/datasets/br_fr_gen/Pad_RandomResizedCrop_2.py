import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(29, 117, 176), padding_mode='symmetric'),
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.87), ratio=(0.9, 2.73)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
