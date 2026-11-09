import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.Grayscale(num_output_channels=3),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.34, 1.62)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
