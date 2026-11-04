import torch<br/>
import torchvision.transforms as transforms<br/><br/>
def transform(norm):<br/>
&nbsp; &nbsp; return transforms.Compose([<br/>
&nbsp; &nbsp; &nbsp; transforms.Resize(256),<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomRotation(10),<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomCrop(224),<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomHorizontalFlip(p=0.5),<br/>
&nbsp; &nbsp; &nbsp; transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br/>
&nbsp; &nbsp; &nbsp; transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=0.5),<br/>
&nbsp; &nbsp; &nbsp; transforms.ToTensor(),<br/>
&nbsp; &nbsp; &nbsp; transforms.Normalize(**norm)<br/>
&nbsp; &nbsp; ])