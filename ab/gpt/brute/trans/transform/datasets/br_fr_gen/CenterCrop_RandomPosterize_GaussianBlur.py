import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=26),
    transforms.RandomPosterize(bits=7, p=0.61),
    transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 1.28)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
