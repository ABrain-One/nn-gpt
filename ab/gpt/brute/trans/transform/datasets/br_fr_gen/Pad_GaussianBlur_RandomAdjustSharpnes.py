import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(79, 251, 155), padding_mode='reflect'),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.58, 1.73)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.16, p=0.22),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
