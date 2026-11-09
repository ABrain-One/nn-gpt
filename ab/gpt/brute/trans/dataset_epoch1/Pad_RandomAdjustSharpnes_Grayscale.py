import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(81, 184, 221), padding_mode='symmetric'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.88, p=0.52),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
