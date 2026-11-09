import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.01, contrast=0.91, saturation=1.11, hue=0.09),
    transforms.CenterCrop(size=31),
    transforms.RandomAdjustSharpness(sharpness_factor=1.22, p=0.67),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
