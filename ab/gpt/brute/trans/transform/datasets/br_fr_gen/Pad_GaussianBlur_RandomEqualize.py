import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(73, 255, 35), padding_mode='edge'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.52, 1.87)),
    transforms.RandomEqualize(p=0.36),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
