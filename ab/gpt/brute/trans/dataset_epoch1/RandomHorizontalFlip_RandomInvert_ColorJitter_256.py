import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.39),
    transforms.RandomInvert(p=0.88),
    transforms.ColorJitter(brightness=0.91, contrast=0.82, saturation=1.2, hue=0.02),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
