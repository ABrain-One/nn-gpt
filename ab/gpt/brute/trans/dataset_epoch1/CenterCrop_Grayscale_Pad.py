import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=24),
    transforms.Grayscale(num_output_channels=3),
    transforms.Pad(padding=4, fill=(80, 45, 241), padding_mode='edge'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
