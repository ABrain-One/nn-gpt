import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.62),
    transforms.RandomRotation(degrees=3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.78, 1.82)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
