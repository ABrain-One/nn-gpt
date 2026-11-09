import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=27),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.74, 1.09)),
    transforms.RandomRotation(degrees=2),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
