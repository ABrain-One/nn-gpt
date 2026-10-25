import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(160, 191, 128), padding_mode='symmetric'),
    transforms.ColorJitter(brightness=0.96, contrast=1.14, saturation=0.96, hue=0.08),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.53, 1.29)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
