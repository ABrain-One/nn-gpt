import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.17, contrast=1.09, saturation=1.18, hue=0.06),
    transforms.RandomCrop(size=27),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.24, 1.25)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
