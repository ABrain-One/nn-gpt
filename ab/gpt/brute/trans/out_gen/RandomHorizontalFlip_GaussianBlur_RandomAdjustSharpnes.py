import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.34),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.62, 1.41)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.18, p=0.89),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
