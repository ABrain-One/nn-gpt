import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.8), ratio=(0.95, 2.81)),
    transforms.Pad(padding=1, fill=(40, 97, 213), padding_mode='symmetric'),
    transforms.RandomVerticalFlip(p=0.12),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
