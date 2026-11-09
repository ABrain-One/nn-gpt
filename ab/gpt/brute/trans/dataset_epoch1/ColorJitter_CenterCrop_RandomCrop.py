import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.96, contrast=1.05, saturation=0.98, hue=0.05),
    transforms.CenterCrop(size=29),
    transforms.RandomCrop(size=29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
