import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.95), ratio=(0.84, 1.39)),
    transforms.Pad(padding=5, fill=(148, 91, 169), padding_mode='reflect'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.73, 1.73)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
