import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.76),
    transforms.RandomAffine(degrees=19, translate=(0.08, 0.1), scale=(1.08, 1.92), shear=(2.57, 9.22)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.41, 1.34)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
