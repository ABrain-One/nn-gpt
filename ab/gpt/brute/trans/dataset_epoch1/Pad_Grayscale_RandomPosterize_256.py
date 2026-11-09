import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(226, 202, 142), padding_mode='constant'),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomPosterize(bits=8, p=0.2),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
