import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.ColorJitter(brightness=1.03, contrast=0.94, saturation=0.93, hue=0.08),
    transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.75),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
