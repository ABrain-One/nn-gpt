import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.89), ratio=(1.31, 1.46)),
    transforms.ColorJitter(brightness=1.17, contrast=0.94, saturation=0.91, hue=0.02),
    transforms.RandomPosterize(bits=5, p=0.68),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
