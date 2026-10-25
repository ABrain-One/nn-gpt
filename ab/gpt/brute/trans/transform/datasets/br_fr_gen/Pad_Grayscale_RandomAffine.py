import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(105, 8, 169), padding_mode='symmetric'),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomAffine(degrees=10, translate=(0.06, 0.19), scale=(1.12, 1.63), shear=(1.88, 6.55)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
