import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

from Mstyle_transfer import args
from models import VGG
from utils import make_transform, load_image, save_image, calculate_style_loss, calculate_content_loss, save_image1
import csv
import os
import json
from PIL import Image


def load_image(image_path, transform, grayscale=False):
    if grayscale:
        image = Image.open(image_path).convert('L')  # 转换为灰度
    else:
        image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image


# 内容图像和风格图像的转换（包含标准化）
content_style_transform = make_transform(args.image_size, normalize=True, grayscale=False)
# 生成图像的转换（不标准化，转换为灰度）
generated_transform = make_transform(args.image_size, normalize=False, grayscale=True)

if args.init_image_flag:
    init_image = load_image(args.init_image, generated_transform, grayscale=True).to(device)
else:
    # 创建单通道噪声图像
    noise_array = np.random.randint(0, 256, args.image_size, dtype=np.uint8)
    noise_image = Image.fromarray(noise_array, mode='L')
    noise_image_path = os.path.join(args.output_dir, "noise_image.jpg")
    noise_image.save(noise_image_path)
    init_image = load_image(noise_image_path, generated_transform, grayscale=True).to(device)

generated_img = init_image.clone().requires_grad_(True)

def vgg_model(generated_input):
    pass


# 扩展为三通道并应用VGG标准化参数
normalizer = torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
generated_input = generated_img.expand(-1, 3, -1, -1)  # 单通道扩展为三通道
generated_input = normalizer(generated_input)
generated_content, generated_style = vgg_model(generated_input)


def save_image(tensor, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    tensor = tensor.cpu().clone().detach().squeeze(0)

    if tensor.dim() == 2 or tensor.size(0) == 1:
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        tensor = (tensor * 255).clamp(0, 255).byte()
        image = Image.fromarray(tensor.numpy(), mode='L')
    else:
        tensor = (tensor * 255).clamp(0, 255).byte()
        image = Image.fromarray(tensor.permute(1, 2, 0).numpy(), 'RGB')

    image.save(os.path.join(output_dir, filename))