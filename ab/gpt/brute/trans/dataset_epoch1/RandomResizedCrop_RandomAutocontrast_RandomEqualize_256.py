import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.65, 0.95), ratio=(0.86, 2.89)),
    transforms.RandomAutocontrast(p=0.28),
    transforms.RandomEqualize(p=0.69),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
