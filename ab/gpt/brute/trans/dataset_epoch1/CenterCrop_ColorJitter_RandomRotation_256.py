import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=31),
    transforms.ColorJitter(brightness=0.88, contrast=1.03, saturation=1.12, hue=0.09),
    transforms.RandomRotation(degrees=20),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
