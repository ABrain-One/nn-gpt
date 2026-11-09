import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.88, 1.83)),
    transforms.RandomCrop(size=25),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
