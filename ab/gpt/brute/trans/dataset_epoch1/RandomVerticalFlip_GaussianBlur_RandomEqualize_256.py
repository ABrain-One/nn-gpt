import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.25),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.84, 1.31)),
    transforms.RandomEqualize(p=0.62),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
