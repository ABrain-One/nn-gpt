import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.44),
    transforms.CenterCrop(size=29),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.48, 1.05)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
