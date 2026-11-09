import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=26),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.19, 1.56)),
    transforms.RandomAffine(degrees=22, translate=(0.19, 0.11), scale=(0.95, 1.57), shear=(2.17, 7.52)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
