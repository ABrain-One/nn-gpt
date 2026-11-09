import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(78, 152, 5), padding_mode='symmetric'),
    transforms.RandomPosterize(bits=5, p=0.75),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
