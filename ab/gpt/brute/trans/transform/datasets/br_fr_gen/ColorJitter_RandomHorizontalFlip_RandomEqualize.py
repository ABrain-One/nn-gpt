import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.86, contrast=1.14, saturation=1.04, hue=0.01),
    transforms.RandomHorizontalFlip(p=0.46),
    transforms.RandomEqualize(p=0.37),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
