import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(166, 10, 112), padding_mode='constant'),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.22, 1.91)),
    transforms.ColorJitter(brightness=1.08, contrast=0.92, saturation=1.03, hue=0.02),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
