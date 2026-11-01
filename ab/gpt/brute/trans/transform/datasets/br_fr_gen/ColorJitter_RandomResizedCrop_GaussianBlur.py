import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.97, contrast=0.88, saturation=0.84, hue=0.01),
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.83), ratio=(0.9, 2.33)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.52, 1.46)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
