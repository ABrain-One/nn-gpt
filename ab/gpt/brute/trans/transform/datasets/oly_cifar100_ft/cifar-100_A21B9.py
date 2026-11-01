import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.62, 0.95), ratio=(0.89, 2.83)), transforms.RandomPerspective(distortion_scale=0.18, p=0.43), transforms.GaussianBlur(kernel_size=5, sigma=(0.87, 1.25)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.481, 0.461, 0.7), (0.288, 0.226, 0.283))])