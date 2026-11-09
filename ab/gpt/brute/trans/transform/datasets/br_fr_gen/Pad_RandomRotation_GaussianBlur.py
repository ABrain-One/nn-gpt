import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(244, 255, 133), padding_mode='symmetric'),
    transforms.RandomRotation(degrees=22),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.32, 1.8)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
