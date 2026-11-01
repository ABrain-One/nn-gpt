import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=24),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.43, 1.37)),
    transforms.RandomEqualize(p=0.18),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
