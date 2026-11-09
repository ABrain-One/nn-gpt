import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.9), ratio=(1.27, 1.93)),
    transforms.RandomRotation(degrees=21),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.25, 1.53)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
