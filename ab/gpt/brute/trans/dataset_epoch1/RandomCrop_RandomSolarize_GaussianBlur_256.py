import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=30),
    transforms.RandomSolarize(threshold=223, p=0.24),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.7, 1.58)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
