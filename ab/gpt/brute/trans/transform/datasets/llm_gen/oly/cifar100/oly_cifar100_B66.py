import torch<br/>
import torchvision.transforms as transforms<br/><br/>
def transform(norm):<br/>
&nbsp;&nbsp;return transforms.Compose([<br/>
&emsp;&emsp;transforms.RandomVerticalFlip(p=0.5),<br/>
&emsp;&emsp;transforms.RandomResizedCrop(64, scale=(0.5, 1.0), ratio=(0.9, 1.1)),<br/>
&emsp;&emsp;transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),<br/>
&emsp;&emsp;transforms.RandomRotation(degrees=(-15, 15)),<br/>
&emsp;&emsp;transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),<br/>
&emsp;&emsp;transforms.ToTensor(),<br/>
&emsp;&emsp;transforms.Normalize(*norm)<br/>
&nbsp;&nbsp;])