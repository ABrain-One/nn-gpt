import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=25),
    transforms.RandomHorizontalFlip(p=0.51),
    transforms.ColorJitter(brightness=0.83, contrast=0.98, saturation=1.07, hue=0.06),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
