import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(51, 67, 101), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.6, p=0.66),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.89, 1.61)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
