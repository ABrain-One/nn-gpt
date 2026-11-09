import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.85), ratio=(0.83, 1.7)),
    transforms.RandomGrayscale(p=0.82),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.88, 1.7)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
