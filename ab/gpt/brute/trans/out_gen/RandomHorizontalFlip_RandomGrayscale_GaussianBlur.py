import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.57),
    transforms.RandomGrayscale(p=0.82),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.71, 1.3)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
