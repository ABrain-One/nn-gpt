import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.84), ratio=(1.01, 1.88)),
    transforms.RandomPerspective(distortion_scale=0.22, p=0.65),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.73, 1.85)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
