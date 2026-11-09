import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.94), ratio=(1.05, 1.83)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.29, 1.46)),
    transforms.ColorJitter(brightness=0.84, contrast=0.88, saturation=1.03, hue=0.0),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
