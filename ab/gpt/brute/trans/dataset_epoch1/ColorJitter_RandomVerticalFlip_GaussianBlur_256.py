import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.08, contrast=1.14, saturation=1.04, hue=0.04),
    transforms.RandomVerticalFlip(p=0.84),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.96, 1.7)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
