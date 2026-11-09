import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(48, 69, 80), padding_mode='reflect'),
    transforms.RandomAutocontrast(p=0.64),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.63, 1.35)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
