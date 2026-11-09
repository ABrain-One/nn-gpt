import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.89), ratio=(1.29, 1.92)),
    transforms.Pad(padding=5, fill=(83, 209, 0), padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(p=0.58),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
