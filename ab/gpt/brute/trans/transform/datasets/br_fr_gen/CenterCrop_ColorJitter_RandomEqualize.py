import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=30),
    transforms.ColorJitter(brightness=0.99, contrast=1.09, saturation=1.09, hue=0.08),
    transforms.RandomEqualize(p=0.12),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
