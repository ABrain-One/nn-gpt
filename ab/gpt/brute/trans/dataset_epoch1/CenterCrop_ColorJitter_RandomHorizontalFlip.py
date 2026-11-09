import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=25),
    transforms.ColorJitter(brightness=1.0, contrast=1.1, saturation=0.97, hue=0.02),
    transforms.RandomHorizontalFlip(p=0.37),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
