import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.73),
    transforms.CenterCrop(size=27),
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.97), ratio=(0.96, 2.92)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
