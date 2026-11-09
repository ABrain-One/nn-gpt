import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(194, 244, 182), padding_mode='edge'), transforms.GaussianBlur(kernel_size=5, sigma=(0.24, 1.33)), transforms.RandomEqualize(p=0.65), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(mean=[0.5198, 0.4209, 0.4986], std=[0.2205, 0.2152, 0.2862])])