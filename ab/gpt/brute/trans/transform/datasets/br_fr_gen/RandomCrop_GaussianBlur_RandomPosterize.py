import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.33, 1.22)),
    transforms.RandomPosterize(bits=4, p=0.49),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
