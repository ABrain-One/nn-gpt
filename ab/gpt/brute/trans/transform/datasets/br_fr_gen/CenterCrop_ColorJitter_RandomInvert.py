import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=32),
    transforms.ColorJitter(brightness=0.84, contrast=0.88, saturation=0.82, hue=0.02),
    transforms.RandomInvert(p=0.78),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
