import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=30),
    transforms.RandomPerspective(distortion_scale=0.12, p=0.48),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.82, 1.49)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
