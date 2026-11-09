import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=32),
    transforms.RandomEqualize(p=0.26),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.64, 1.54)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
