import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.09, contrast=1.12, saturation=1.11, hue=0.1),
    transforms.RandomRotation(degrees=26),
    transforms.RandomSolarize(threshold=29, p=0.67),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
