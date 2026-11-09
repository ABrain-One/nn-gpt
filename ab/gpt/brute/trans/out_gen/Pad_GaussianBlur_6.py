import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(214, 209, 138), padding_mode='constant'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.13, 1.55)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
