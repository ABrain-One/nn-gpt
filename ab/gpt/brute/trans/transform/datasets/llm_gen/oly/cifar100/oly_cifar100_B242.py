import torch<br/>
import torchvision.transforms as transforms<br/><br/>
def transform(norm):<br/>
&nbsp;return transforms.Compose([<br/>
&nbsp;&nbsp;transforms.Resize(256),<br/>
&nbsp;&nbsp;transforms.RandomResizedCrop(224, scale=(0.64, 1.0), ratio=(0.9, 1.1)),<br/>
&nbsp;&nbsp;transforms.RandomHorizontalFlip(p=0.5),<br/>
&nbsp;&nbsp;transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br/>
&nbsp;&nbsp;transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.25),<br/>
&nbsp;&nbsp;transforms.ToTensor(),<br/>
&nbsp;&nbsp;transforms.Normalize(**norm)<br/>
&nbsp;])