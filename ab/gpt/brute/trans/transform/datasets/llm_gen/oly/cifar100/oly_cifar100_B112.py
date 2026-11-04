import torch<br/>
import torchvision.transforms as transforms<br/>
&nbsp;<br/>
def transform(norm):<br/>
&nbsp;return transforms.Compose([<br/>
&nbsp;&nbsp;transforms.RandomResizedCrop(size=32, scale=(0.6, 1.0), ratio=(0.75, 1.33)),<br/>
&nbsp;&nbsp;transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),<br/>
&nbsp;&nbsp;transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=1.0)], p=0.5),<br/>
&nbsp;&nbsp;transforms.RandomHorizontalFlip(p=0.5),<br/>
&nbsp;&nbsp;transforms.Resize(224),<br/>
&nbsp;&nbsp;transforms.ToTensor(),<br/>
&nbsp;&nbsp;transforms.Normalize(*norm)<br/>
&nbsp;])<br/>