import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.44),
    transforms.CenterCrop(size=27),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.92, 1.16)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
