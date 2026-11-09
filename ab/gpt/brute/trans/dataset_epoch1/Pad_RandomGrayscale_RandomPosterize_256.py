import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(149, 238, 150), padding_mode='symmetric'),
    transforms.RandomGrayscale(p=0.62),
    transforms.RandomPosterize(bits=4, p=0.3),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
