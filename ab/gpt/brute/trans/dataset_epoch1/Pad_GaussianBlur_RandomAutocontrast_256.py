import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(42, 144, 150), padding_mode='symmetric'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.71, 1.31)),
    transforms.RandomAutocontrast(p=0.13),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
