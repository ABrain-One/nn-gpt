import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.96, 1.57)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.03, p=0.11),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
