import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.95), ratio=(0.83, 1.77)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.74, 1.76)),
    transforms.RandomSolarize(threshold=31, p=0.69),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
