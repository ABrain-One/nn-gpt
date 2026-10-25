import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(237, 60, 63), padding_mode='reflect'),
    transforms.RandomPosterize(bits=6, p=0.49),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.24, 1.58)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])