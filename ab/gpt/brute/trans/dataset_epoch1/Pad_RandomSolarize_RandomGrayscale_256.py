import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(147, 48, 153), padding_mode='constant'),
    transforms.RandomSolarize(threshold=124, p=0.78),
    transforms.RandomGrayscale(p=0.34),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
