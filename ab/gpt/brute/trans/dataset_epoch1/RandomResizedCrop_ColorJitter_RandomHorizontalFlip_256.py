import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.87), ratio=(0.93, 1.95)),
    transforms.ColorJitter(brightness=0.91, contrast=1.11, saturation=0.88, hue=0.07),
    transforms.RandomHorizontalFlip(p=0.77),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
