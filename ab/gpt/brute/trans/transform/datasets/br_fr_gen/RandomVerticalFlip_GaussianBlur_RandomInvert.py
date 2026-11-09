import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.6),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.81, 1.01)),
    transforms.RandomInvert(p=0.17),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
