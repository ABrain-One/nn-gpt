import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.49, 1.8)),
    transforms.RandomGrayscale(p=0.69),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
