import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.83, contrast=1.0, saturation=0.88, hue=0.09),
    transforms.Pad(padding=4, fill=(228, 155, 46), padding_mode='symmetric'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.29, 1.52)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
