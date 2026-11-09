import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.85),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.6, 1.44)),
    transforms.RandomAffine(degrees=21, translate=(0.07, 0.09), scale=(1.17, 1.88), shear=(1.58, 8.81)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
