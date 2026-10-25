import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.66),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.53, 1.78)),
    transforms.RandomAffine(degrees=0, translate=(0.17, 0.12), scale=(0.91, 1.94), shear=(2.09, 5.47)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
