import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(65, 92, 57), padding_mode='edge'),
    transforms.RandomEqualize(p=0.23),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.91, 1.1)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
