import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.RandomAdjustSharpness(sharpness_factor=1.9, p=0.34),
    transforms.RandomHorizontalFlip(p=0.58),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
