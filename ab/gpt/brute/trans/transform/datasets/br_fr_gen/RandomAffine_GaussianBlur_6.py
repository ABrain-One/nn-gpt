import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=11, translate=(0.1, 0.11), scale=(1.08, 1.76), shear=(3.1, 5.84)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.74, 1.06)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
