import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.55),
    transforms.RandomHorizontalFlip(p=0.53),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.48, 1.13)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
