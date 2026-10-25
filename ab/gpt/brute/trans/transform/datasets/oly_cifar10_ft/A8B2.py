import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.Pad(padding=1, fill=(240, 28, 225), padding_mode='symmetric'),
        transforms.RandomResizedCrop(size=32, scale=(0.6, 0.94), ratio=(1.16, 2.51)),
        transforms.RandomVerticalFlip(p=0.16),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])