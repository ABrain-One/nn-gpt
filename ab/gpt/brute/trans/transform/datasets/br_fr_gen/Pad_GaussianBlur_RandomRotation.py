import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(73, 213, 222), padding_mode='reflect'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.44, 1.45)),
    transforms.RandomRotation(degrees=25),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
