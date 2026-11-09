import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=30),
    transforms.RandomCrop(size=30),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.63, 1.51)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
