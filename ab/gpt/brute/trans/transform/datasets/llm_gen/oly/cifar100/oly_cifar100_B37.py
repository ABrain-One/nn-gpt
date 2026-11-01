import torch<br/>
import torchvision.transforms as transforms<br/><br/>
def transform(norm):<br/>
&nbsp;&nbsp;return transforms.Compose([<br/>
&thinsp;&thinsp;transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),<br/>
&thinsp;&thinsp;transforms.RandomHorizontalFlip(p=0.5),<br/>
&thinsp;&thinsp;transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)),<br/>
&thinsp;&thinsp;transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br/>
&thinsp;&thinsp;transforms.ToTensor(),<br/>
&thinsp;&thinsp;transforms.Normalize(*norm)<br/>
&nbsp;&nbsp;])<br/>