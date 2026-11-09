import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(134, 119, 81), padding_mode='reflect'),
    transforms.CenterCrop(size=29),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.9, 1.0)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
