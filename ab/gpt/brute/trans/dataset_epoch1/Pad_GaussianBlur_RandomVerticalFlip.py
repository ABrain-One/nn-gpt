import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(24, 145, 20), padding_mode='symmetric'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.57, 1.05)),
    transforms.RandomVerticalFlip(p=0.38),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
