import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.53),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.93, 1.73)),
    transforms.RandomEqualize(p=0.86),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
