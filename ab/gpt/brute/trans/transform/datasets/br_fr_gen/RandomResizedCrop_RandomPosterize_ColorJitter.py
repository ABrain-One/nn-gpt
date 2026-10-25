import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.85), ratio=(0.99, 1.69)),
    transforms.RandomPosterize(bits=4, p=0.63),
    transforms.ColorJitter(brightness=1.1, contrast=1.14, saturation=0.98, hue=0.08),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
