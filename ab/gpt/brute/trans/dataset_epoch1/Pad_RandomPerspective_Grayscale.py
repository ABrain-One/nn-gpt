import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(112, 253, 165), padding_mode='reflect'),
    transforms.RandomPerspective(distortion_scale=0.17, p=0.81),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
