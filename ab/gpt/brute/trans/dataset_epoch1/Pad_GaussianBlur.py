import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(181, 117, 151), padding_mode='edge'),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.72, 1.9)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
