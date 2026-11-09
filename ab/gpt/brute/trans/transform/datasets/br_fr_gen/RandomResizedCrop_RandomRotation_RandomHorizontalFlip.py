import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.89), ratio=(1.26, 2.48)),
    transforms.RandomRotation(degrees=17),
    transforms.RandomHorizontalFlip(p=0.18),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
