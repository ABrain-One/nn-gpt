import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.5, 0.93), ratio=(0.91, 1.39)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.98, 1.93)),
    transforms.RandomGrayscale(p=0.26),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
