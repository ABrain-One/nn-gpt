import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=28),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.29, 1.39)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.43, p=0.82),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
