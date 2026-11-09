import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=30),
    transforms.RandomVerticalFlip(p=0.19),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.57, 1.26)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
