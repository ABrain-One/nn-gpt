import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.96), ratio=(1.0, 1.83)),
    transforms.RandomSolarize(threshold=158, p=0.37),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.81, 1.25)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
