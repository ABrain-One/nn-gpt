import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.RandomAffine(degrees=6, translate=(0.1, 0.2), scale=(0.85, 1.45), shear=(0.02, 5.66)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.24, 1.55)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
