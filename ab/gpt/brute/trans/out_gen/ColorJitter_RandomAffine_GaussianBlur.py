import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.04, contrast=1.01, saturation=0.94, hue=0.01),
    transforms.RandomAffine(degrees=29, translate=(0.11, 0.04), scale=(0.91, 1.86), shear=(1.8, 8.77)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.63, 1.02)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
