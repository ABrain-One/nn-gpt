import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.52),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 1.03)),
    transforms.RandomCrop(size=24),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
