import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(121, 83, 208), padding_mode='symmetric'),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomInvert(p=0.25),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
