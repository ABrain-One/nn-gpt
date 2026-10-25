import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=30),
    transforms.ColorJitter(brightness=0.88, contrast=1.19, saturation=0.91, hue=0.07),
    transforms.RandomAdjustSharpness(sharpness_factor=1.45, p=0.37),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
