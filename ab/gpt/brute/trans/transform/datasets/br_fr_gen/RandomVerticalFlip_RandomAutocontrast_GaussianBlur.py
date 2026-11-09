import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.66),
    transforms.RandomAutocontrast(p=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.25, 1.99)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
