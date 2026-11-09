import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.55),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.52, 1.48)),
    transforms.ColorJitter(brightness=0.81, contrast=0.91, saturation=0.91, hue=0.0),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
