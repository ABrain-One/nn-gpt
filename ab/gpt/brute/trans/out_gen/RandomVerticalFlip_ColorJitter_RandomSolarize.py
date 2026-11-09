import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.26),
    transforms.ColorJitter(brightness=1.1, contrast=1.19, saturation=0.88, hue=0.04),
    transforms.RandomSolarize(threshold=97, p=0.87),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
