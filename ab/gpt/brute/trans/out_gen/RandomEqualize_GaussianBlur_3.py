import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.29),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.86, 1.1)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
