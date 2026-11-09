import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=31),
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.96), ratio=(1.02, 1.51)),
    transforms.RandomHorizontalFlip(p=0.69),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
