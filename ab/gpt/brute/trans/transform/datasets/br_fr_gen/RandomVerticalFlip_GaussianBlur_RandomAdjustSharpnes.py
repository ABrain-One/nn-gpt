import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.65),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.91, 1.58)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.93, p=0.29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
