import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.79),
    transforms.RandomEqualize(p=0.26),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.57, 1.93)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])