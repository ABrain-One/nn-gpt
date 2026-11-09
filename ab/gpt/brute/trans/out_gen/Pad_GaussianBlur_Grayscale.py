import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(2, 210, 234), padding_mode='edge'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.87, 1.84)),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
