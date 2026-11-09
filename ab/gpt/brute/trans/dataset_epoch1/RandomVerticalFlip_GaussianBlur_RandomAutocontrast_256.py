import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.64),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.86, 1.34)),
    transforms.RandomAutocontrast(p=0.69),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
