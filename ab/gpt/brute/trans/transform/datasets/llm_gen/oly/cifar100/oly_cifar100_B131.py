import torch<br/>
import torchvision.transforms as transforms<br/>
<br/>
def transform(norm):<br/>
&nbsp;&nbsp;return transforms.Compose([<br/>
&nbsp;&nbsp;transforms.Pad(padding=4, fill=(199/255.0, 77/255.0, 171/255.0), padding_mode='symmetric'),<br/>
&nbsp;&nbsp;transforms.RandomCrop(32),<br/>
&nbsp;&nbsp;transforms.RandomHorizontalFlip(p=0.5),<br/>
&nbsp;&nbsp;transforms.RandomVerticalFlip(p=0.1),<br/>
&nbsp;&nbsp;transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br/>
&nbsp;&nbsp;transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),<br/>
&nbsp;&nbsp;transforms.Resize(224),<br/>
&nbsp;&nbsp;transforms.ToTensor(),<br/>
&nbsp;&nbsp;transforms.Normalize(**norm),<br/>
&nbsp;&nbsp;])<br/>