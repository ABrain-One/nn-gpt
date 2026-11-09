import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=29),
    transforms.RandomEqualize(p=0.68),
    transforms.ColorJitter(brightness=0.95, contrast=0.98, saturation=0.87, hue=0.02),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
