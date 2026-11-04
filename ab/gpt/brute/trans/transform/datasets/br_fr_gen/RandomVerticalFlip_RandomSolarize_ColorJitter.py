import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.52),
    transforms.RandomSolarize(threshold=225, p=0.67),
    transforms.ColorJitter(brightness=1.11, contrast=1.11, saturation=0.88, hue=0.05),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
