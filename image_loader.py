from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_cuda = torch.cuda.is_available()

imsize = 512 if use_cuda else 128

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])


def load_image(img_path):
    image = Image.open(img_path)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)