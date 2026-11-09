import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.94), ratio=(1.22, 1.62)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.95, 1.57)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.08, p=0.43),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
