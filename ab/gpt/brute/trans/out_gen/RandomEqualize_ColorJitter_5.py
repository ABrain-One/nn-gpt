import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.64),
    transforms.ColorJitter(brightness=0.89, contrast=0.87, saturation=0.88, hue=0.06),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
