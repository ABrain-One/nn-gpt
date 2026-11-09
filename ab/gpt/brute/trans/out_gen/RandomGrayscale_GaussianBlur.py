import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.14),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.41, 1.87)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
