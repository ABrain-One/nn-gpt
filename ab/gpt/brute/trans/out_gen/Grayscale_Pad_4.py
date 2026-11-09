import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Pad(padding=4, fill=(141, 13, 225), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
