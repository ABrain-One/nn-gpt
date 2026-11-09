import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.44),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.42, 1.68)),
    transforms.RandomGrayscale(p=0.35),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
