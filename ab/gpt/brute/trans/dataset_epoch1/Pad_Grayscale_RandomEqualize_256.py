import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(149, 171, 0), padding_mode='symmetric'),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomEqualize(p=0.84),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
