import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.53),
    transforms.ColorJitter(brightness=0.91, contrast=0.93, saturation=0.99, hue=0.08),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.79, 1.42)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
