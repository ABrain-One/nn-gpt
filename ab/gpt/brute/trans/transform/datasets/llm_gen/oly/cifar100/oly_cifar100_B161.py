import torch<br/>
import torchvision.transforms as transforms<br/>
<br/>
def transform(norm):<br/>
    return transforms.Compose([<br/>
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br/>
        transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0)),<br/>
        transforms.RandomHorizontalFlip(p=0.5),<br/>
        transforms.RandomRotation(degrees=(-10, 10)),<br/>
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),<br/>
        transforms.Resize((224, 224)),<br/>
        transforms.ToTensor(),<br/>
        transforms.Normalize(**norm)<br/>
    ])