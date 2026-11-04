import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.RandomAdjustSharpness(sharpness_factor=1.47, p=0.25),
    transforms.ColorJitter(brightness=1.16, contrast=0.94, saturation=0.91, hue=0.08),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
