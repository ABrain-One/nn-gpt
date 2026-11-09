import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.78, 1.08)),
    transforms.Pad(padding=3, fill=(230, 215, 105), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
