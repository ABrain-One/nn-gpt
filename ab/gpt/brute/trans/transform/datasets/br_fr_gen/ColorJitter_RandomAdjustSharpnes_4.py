import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.07, contrast=1.05, saturation=0.95, hue=0.02),
    transforms.RandomAdjustSharpness(sharpness_factor=1.76, p=0.55),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
