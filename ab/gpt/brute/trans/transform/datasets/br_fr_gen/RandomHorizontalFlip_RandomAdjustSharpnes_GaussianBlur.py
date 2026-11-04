import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.73),
    transforms.RandomAdjustSharpness(sharpness_factor=1.59, p=0.23),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.74, 1.36)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
