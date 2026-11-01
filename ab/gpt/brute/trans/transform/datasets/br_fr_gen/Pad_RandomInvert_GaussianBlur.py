import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(165, 82, 112), padding_mode='edge'),
    transforms.RandomInvert(p=0.21),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.46, 1.7)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
