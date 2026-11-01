import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=31),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.46, 1.13)),
    transforms.RandomInvert(p=0.15),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
