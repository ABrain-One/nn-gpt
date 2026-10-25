import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(157, 186, 122), padding_mode='reflect'),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.86, 1.74)),
    transforms.RandomHorizontalFlip(p=0.71),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
