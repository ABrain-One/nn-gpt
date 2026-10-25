import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(0, 233, 113), padding_mode='symmetric'),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomVerticalFlip(p=0.65),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
