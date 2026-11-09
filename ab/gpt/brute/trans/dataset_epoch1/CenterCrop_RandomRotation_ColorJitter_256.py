import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=30),
    transforms.RandomRotation(degrees=19),
    transforms.ColorJitter(brightness=1.09, contrast=1.2, saturation=1.14, hue=0.09),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
