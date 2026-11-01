import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.77),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.78, 1.41)),
    transforms.RandomRotation(degrees=28),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
