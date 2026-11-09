import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=25),
    transforms.CenterCrop(size=30),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.79, 1.39)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
