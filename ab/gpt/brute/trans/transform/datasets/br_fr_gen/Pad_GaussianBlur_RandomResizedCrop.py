import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(133, 234, 249), padding_mode='symmetric'),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.79, 1.47)),
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.92), ratio=(1.01, 2.43)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
