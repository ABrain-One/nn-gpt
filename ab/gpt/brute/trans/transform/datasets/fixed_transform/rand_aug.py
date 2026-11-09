import torchvision.transforms as T

def transform(norm):
    return T.Compose([
    T.TrivialAugmentWide(),
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(*norm)
])