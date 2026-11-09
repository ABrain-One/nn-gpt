import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(211, 89, 249), padding_mode='edge'),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.81),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
