import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.66, 1.28)),
    transforms.ColorJitter(brightness=0.94, contrast=0.91, saturation=1.0, hue=0.09),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
