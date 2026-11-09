import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=7, translate=(0.17, 0.07), scale=(1.03, 1.54), shear=(3.92, 8.58)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.23, 1.26)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
