import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.68),
    transforms.RandomPosterize(bits=4, p=0.26),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.39, 1.08)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
