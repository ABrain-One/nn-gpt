import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(181, 202, 233), padding_mode='edge'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.14, 1.67)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
