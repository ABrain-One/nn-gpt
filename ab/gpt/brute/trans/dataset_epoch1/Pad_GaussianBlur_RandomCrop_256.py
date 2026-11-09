import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(82, 190, 204), padding_mode='edge'),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.35, 1.49)),
    transforms.RandomCrop(size=30),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
