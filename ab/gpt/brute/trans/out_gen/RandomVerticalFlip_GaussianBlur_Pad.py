import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.6),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.27, 1.9)),
    transforms.Pad(padding=3, fill=(60, 214, 148), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
