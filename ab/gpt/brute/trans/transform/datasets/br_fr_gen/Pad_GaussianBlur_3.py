import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(31, 146, 211), padding_mode='edge'),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 1.42)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
