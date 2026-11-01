import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.RandomPosterize(bits=6, p=0.83),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.21, 1.37)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
