import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.04, contrast=1.04, saturation=1.17, hue=0.04),
    transforms.RandomGrayscale(p=0.66),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.29, 1.26)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
