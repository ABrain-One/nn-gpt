import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.35),
    transforms.ColorJitter(brightness=0.88, contrast=1.15, saturation=0.95, hue=0.05),
    transforms.RandomSolarize(threshold=207, p=0.66),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
