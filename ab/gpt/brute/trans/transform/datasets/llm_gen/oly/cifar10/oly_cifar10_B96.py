import torch<br/>
import torchvision.transforms as T<br/><br/>
def transform(norm):<br/>
&nbsp;&nbsp;return T.Compose([<br/>
&thinsp;&thinsp;T.RandomRotation(15),<br/>
&thinsp;&thinsp;T.RandomHorizontalFlip(p=0.5),<br/>
&thinsp;&thinsp;T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br/>
&thinsp;&thinsp;T.RandomAffine(<br/>
&thinsp;&thinsp;&thinsp;degrees=15,<br/>
&thinsp;&thinsp;&thinsp;translate=(0.2, 0.2),<br/>
&thinsp;&thinsp;&thinsp;scale=(0.8, 1.2),<br/>
&thinsp;&thinsp;&thinsp;shear=(-5, 5),<br/>
&thinsp;&thinsp;&thinsp;fill=(0.4914, 0.4822, 0.4465)<br/>
&thinsp;&thinsp;),<br/>
&thinsp;T.Resize(224),<br/>
&thinsp;T.ToTensor(),<br/>
&thinsp;T.Normalize(**norm)<br/>
&thinsp;])<br/>