import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.98), ratio=(0.86, 1.99)),
    transforms.RandomSolarize(threshold=213, p=0.57),
    transforms.ColorJitter(brightness=1.02, contrast=1.13, saturation=0.98, hue=0.03),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
