import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.37),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.64, 1.14)),
    transforms.RandomPosterize(bits=6, p=0.65),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
