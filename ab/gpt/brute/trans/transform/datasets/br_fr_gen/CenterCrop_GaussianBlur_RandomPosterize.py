import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.8, 1.42)),
    transforms.RandomPosterize(bits=6, p=0.12),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
