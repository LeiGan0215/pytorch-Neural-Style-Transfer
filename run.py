from image_loader import load_image
from style_transfer import run_style_transfer
from load_vgg19 import cnn
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

unloader = transforms.ToPILImage()

style_img = load_image("./images/style.jpg")
content_img = load_image("./images/content.jpg")

input_img = content_img.clone()

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

image = output.cpu().clone()  # we clone the tensor to not do changes on it
image = image.squeeze(0)      # remove the fake batch dimension
image = unloader(image)
os.chdir('output')
image.save("output.png")


