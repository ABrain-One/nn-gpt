import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.74),
    transforms.RandomSolarize(threshold=176, p=0.41),
    transforms.ColorJitter(brightness=0.95, contrast=1.1, saturation=0.83, hue=0.04),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
