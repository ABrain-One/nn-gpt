import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=0.85, p=0.27),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.79, 1.17)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
