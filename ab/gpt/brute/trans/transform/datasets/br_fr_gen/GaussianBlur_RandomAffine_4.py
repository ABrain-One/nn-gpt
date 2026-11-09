import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.34, 1.47)),
    transforms.RandomAffine(degrees=3, translate=(0.06, 0.02), scale=(1.15, 1.9), shear=(3.34, 6.48)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
