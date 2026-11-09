import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.88, contrast=0.91, saturation=1.11, hue=0.01),
    transforms.CenterCrop(size=28),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.11, 1.6)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
