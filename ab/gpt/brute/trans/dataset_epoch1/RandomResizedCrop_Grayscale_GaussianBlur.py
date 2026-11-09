import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.5, 0.89), ratio=(0.94, 2.85)),
    transforms.Grayscale(num_output_channels=3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.39, 1.59)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
