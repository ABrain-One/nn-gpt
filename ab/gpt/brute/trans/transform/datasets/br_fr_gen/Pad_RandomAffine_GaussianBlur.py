import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(99, 35, 49), padding_mode='reflect'),
    transforms.RandomAffine(degrees=17, translate=(0.09, 0.11), scale=(1.19, 1.32), shear=(3.17, 6.13)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.39, 1.13)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
