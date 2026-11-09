import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.54, 1.87)),
    transforms.RandomAffine(degrees=6, translate=(0.04, 0.1), scale=(0.92, 1.94), shear=(1.3, 6.55)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
