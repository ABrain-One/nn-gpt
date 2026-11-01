<td>
import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.98, contrast=1.09, saturation=0.97, hue=0.03),
    transforms.RandomRotation(degrees=27),
    transforms.RandomPerspective(distortion_scale=0.13, p=0.50),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
    </td>