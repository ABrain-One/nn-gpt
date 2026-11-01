import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.77),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.75, 1.49)),
    transforms.ColorJitter(brightness=0.84, contrast=1.06, saturation=1.12, hue=0.07),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
