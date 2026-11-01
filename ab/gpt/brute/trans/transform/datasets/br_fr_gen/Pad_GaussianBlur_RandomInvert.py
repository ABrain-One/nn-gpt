import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(82, 254, 33), padding_mode='constant'),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.69, 1.46)),
    transforms.RandomInvert(p=0.81),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
