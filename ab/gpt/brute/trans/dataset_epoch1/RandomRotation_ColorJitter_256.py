import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=19),
    transforms.ColorJitter(brightness=1.15, contrast=1.14, saturation=1.02, hue=0.02),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
