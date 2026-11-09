import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.ColorJitter(brightness=1.0, contrast=1.08, saturation=0.97, hue=0.0),
    transforms.RandomEqualize(p=0.89),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
