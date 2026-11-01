import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.18),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.73, 1.52)),
    transforms.Pad(padding=3, fill=(230, 1, 29), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
