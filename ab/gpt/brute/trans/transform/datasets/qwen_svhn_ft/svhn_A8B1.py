import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.GaussianBlur(kernel_size=5, sigma=(0.18, 1.51)), transforms.RandomEqualize(p=0.17), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])