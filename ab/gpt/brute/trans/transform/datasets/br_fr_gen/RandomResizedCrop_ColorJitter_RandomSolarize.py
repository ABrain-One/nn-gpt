import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.51, 0.83), ratio=(1.27, 1.43)),
    transforms.ColorJitter(brightness=1.13, contrast=1.15, saturation=0.83, hue=0.01),
    transforms.RandomSolarize(threshold=167, p=0.2),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
