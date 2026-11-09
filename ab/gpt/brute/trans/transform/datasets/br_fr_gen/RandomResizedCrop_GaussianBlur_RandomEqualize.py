import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.86), ratio=(0.77, 1.67)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.68, 1.47)),
    transforms.RandomEqualize(p=0.81),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
