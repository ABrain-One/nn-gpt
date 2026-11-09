import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(37, 245, 126), padding_mode='symmetric'),
    transforms.RandomPosterize(bits=6, p=0.79),
    transforms.RandomAffine(degrees=17, translate=(0.08, 0.09), scale=(1.19, 1.61), shear=(2.14, 5.48)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])