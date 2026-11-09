import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.49),
    transforms.RandomPosterize(bits=4, p=0.38),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.86, 1.99)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
