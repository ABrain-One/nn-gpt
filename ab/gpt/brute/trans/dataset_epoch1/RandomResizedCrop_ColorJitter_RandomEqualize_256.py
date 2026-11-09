import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.83), ratio=(1.04, 1.67)),
    transforms.ColorJitter(brightness=1.05, contrast=0.99, saturation=0.84, hue=0.09),
    transforms.RandomEqualize(p=0.45),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
