import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.17, contrast=1.13, saturation=1.17, hue=0.09),
    transforms.RandomCrop(size=31),
    transforms.RandomAdjustSharpness(sharpness_factor=0.55, p=0.19),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
