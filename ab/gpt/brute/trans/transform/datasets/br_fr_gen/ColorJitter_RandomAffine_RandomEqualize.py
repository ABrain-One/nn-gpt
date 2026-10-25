import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.16, contrast=1.18, saturation=0.95, hue=0.01),
    transforms.RandomAffine(degrees=4, translate=(0.16, 0.06), scale=(1.05, 1.72), shear=(2.12, 8.23)),
    transforms.RandomEqualize(p=0.35),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
