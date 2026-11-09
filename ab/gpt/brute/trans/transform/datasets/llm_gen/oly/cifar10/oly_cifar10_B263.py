tag:

< tr>
import torch<br/>
import torchvision.transforms as transforms<br/><br/>
def transform(norm):<br/>&#10; return transforms.Compose([<br/>&#10; transforms.Pad(padding=2, fill=(211, 89, 249), padding_mode='constant'),<br/>&#10; transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(0.9, 1.1)),<br/>&#10; transforms.RandomGrayscale(p=0.25),<br/>&#10; transforms.RandomApply([transforms.HorizontalFlip()], p=0.5),<br/>&#10; transforms.RandomApply([transforms.VerticalFlip()], p=0.2),<br/>&#10; transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br/>&#10; transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),<br/>&#10; transforms.Resize(64),<br/>&#10; transforms.ToTensor(),<br/>&#10; transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994])<br/>&#10; ])<br/>&#10;