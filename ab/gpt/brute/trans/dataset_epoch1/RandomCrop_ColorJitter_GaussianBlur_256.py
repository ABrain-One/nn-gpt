import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.ColorJitter(brightness=0.91, contrast=1.02, saturation=0.97, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.62, 1.72)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
