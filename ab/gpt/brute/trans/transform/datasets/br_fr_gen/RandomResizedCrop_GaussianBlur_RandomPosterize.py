import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.62, 0.99), ratio=(1.31, 2.75)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.13, 1.33)),
    transforms.RandomPosterize(bits=8, p=0.88),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
