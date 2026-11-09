import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(13, 107, 34), padding_mode='edge'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.29, 1.27)),
    transforms.RandomGrayscale(p=0.34),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
