import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.19),
    transforms.Pad(padding=5, fill=(63, 122, 117), padding_mode='constant'),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.98, 1.73)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
