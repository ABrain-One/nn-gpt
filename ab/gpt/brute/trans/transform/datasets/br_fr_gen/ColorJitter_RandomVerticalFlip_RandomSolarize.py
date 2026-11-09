import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.14, contrast=1.09, saturation=1.03, hue=0.03),
    transforms.RandomVerticalFlip(p=0.48),
    transforms.RandomSolarize(threshold=10, p=0.29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
