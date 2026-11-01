import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=25),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.68, 1.34)),
    transforms.RandomRotation(degrees=1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
