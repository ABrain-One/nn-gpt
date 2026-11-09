import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.38),
    transforms.ColorJitter(brightness=1.07, contrast=1.09, saturation=0.96, hue=0.06),
    transforms.RandomEqualize(p=0.87),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
