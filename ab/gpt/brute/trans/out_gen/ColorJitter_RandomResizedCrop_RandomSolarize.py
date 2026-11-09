import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.94, contrast=0.9, saturation=1.2, hue=0.08),
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.92), ratio=(1.16, 2.23)),
    transforms.RandomSolarize(threshold=40, p=0.17),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
