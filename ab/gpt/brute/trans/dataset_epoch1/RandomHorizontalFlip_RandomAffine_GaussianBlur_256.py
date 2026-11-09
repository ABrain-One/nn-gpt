import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.31),
    transforms.RandomAffine(degrees=16, translate=(0.14, 0.09), scale=(0.9, 1.62), shear=(2.34, 7.66)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.83, 1.3)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
