import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.91, 1.02)),
    transforms.RandomGrayscale(p=0.6),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
