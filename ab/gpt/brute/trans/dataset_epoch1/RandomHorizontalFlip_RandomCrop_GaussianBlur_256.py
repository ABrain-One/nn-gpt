import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.81),
    transforms.RandomCrop(size=26),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.52, 1.87)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
