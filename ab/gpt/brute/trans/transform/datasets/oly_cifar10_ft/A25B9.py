import torch
import torchvision.transforms as transforms

def transform(image):
    transformed = transforms.Compose([
    transforms.Pad(padding=5, fill=(161, 176, 128), padding_mode='symmetric'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.23, p=0.76),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48], std=[0.294], max_pixel_value=255.08),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])(
    image
)
return transformed