import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Grayscale(num_output_channels=3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.32, 2.0)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
