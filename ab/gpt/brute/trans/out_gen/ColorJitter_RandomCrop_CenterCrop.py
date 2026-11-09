import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.07, contrast=0.83, saturation=0.92, hue=0.07),
    transforms.RandomCrop(size=29),
    transforms.CenterCrop(size=29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
