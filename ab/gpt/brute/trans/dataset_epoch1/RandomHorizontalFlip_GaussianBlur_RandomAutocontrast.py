import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.68),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.46, 1.44)),
    transforms.RandomAutocontrast(p=0.61),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
