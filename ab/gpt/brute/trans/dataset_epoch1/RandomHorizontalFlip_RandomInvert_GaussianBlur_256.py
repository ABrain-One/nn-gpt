import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.76),
    transforms.RandomInvert(p=0.68),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.76, 1.29)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
