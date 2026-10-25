import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(2, 75, 161), padding_mode='edge'),
    transforms.RandomHorizontalFlip(p=0.78),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.24, 1.19)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
