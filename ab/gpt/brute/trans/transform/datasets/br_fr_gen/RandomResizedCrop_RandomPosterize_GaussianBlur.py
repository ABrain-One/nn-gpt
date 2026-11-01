import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.95), ratio=(1.13, 2.91)),
    transforms.RandomPosterize(bits=8, p=0.28),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.51, 1.91)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
