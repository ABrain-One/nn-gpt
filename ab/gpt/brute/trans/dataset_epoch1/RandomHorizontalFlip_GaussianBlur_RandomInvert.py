import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.81),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.89, 1.64)),
    transforms.RandomInvert(p=0.84),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
