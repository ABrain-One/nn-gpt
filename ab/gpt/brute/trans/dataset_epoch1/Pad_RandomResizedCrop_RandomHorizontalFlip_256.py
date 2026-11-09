import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(255, 79, 204), padding_mode='reflect'),
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.97), ratio=(1.16, 2.93)),
    transforms.RandomHorizontalFlip(p=0.76),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
