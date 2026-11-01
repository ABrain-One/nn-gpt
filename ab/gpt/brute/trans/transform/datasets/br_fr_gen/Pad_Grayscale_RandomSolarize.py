import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(49, 95, 70), padding_mode='edge'),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomSolarize(threshold=76, p=0.74),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
