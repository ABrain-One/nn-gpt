import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=30),
    transforms.RandomAdjustSharpness(sharpness_factor=1.44, p=0.52),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.32, 1.53)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
