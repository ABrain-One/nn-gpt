import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(84, 45, 117), padding_mode='symmetric'),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(degrees=2),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
