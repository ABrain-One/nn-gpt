import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.45),
    transforms.ColorJitter(brightness=0.93, contrast=0.97, saturation=1.05, hue=0.07),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.61, 1.42)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
