import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from models import VGG
from utils import make_transform, load_image, save_image, calculate_style_loss, calculate_content_loss, save_image1
import csv
import os
import json
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='风格迁移算法 - StyleFusion')
    parser.add_argument('--content_image', type=str, default='data/c5.jpg', help='内容图像路径')
    parser.add_argument('--style_image', type=str, default='data/s5.jpg', help='风格图像路径')
    parser.add_argument('--output_dir', type=str, default='./output/01', help='输出目录')
    parser.add_argument("--init_image", type=str, default='', help='初始生成图片')
    parser.add_argument("--init_image_flag", type=bool, default=0, help='是否使用初始图片')
    parser.add_argument('--image_size', type=int, nargs=2, default=[500, 500], help='图像尺寸 (高度, 宽度)')
    parser.add_argument('--content_weight', type=float, default=1, help='内容权重')
    parser.add_argument('--style_weight', type=float, default=30, help='风格权重')
    parser.add_argument('--epochs', type=int, default=500, help='训练周期数')
    parser.add_argument('--steps_per_epoch', type=int, default=50, help='每个周期的步数')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='学习率')

    # 解析参数后立即保存
    m_args = parser.parse_args()
    save_training_config(m_args)  # 新增保存配置功能
    return m_args


def save_training_config(args):
    """保存训练配置到输出目录"""
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    times_tamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(args.output_dir, f"config_{times_tamp}.json")
    config_dict = vars(args)
    config_dict.update({
        "save_time": datetime.now().isoformat(),
        "command_line": " ".join(f"--{k}={v}" for k, v in config_dict.items())
    })

    # 写入JSON文件
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"训练配置已保存至：{config_path}")


args = parse_args()
# 创建带时间戳的唯一文件名
log_dir = os.path.join(args.output_dir, "loss_logs")
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"training_loss_{timestamp}.csv"
filepath = os.path.join(log_dir, filename)
# 初始化CSV文件
with open(filepath, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "style_loss", "content_loss", "total_loss"])

# 内容特征层及loss加权系数
content_layers = {'5': 0.5, '10': 0.5}
# 风格特征层及loss加权系数
style_layers = {'0': 0.2, '5': 0.2, '10': 0.2, '19': 0.2, '28': 0.2}
style_losses = []
content_losses = []
total_losses = []


# ----------------训练即推理过程----------------
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = make_transform(args.image_size, normalize=True)

    content_img = load_image(args.content_image, transform).to(device)  # 读取内容图像
    style_img = load_image(args.style_image, transform).to(device)  # 读取风格图像

    if args.init_image_flag:
        init_image = load_image(args.init_image, transform).to(device)  # 加载并预处理
    else:
        noise_array = np.random.randint(0, 256, args.image_size, dtype=np.uint8)
        noise_image = Image.fromarray(noise_array, mode='L')
        noise_image_path = os.path.join(args.output_dir, "noise_image.jpg")
        noise_image.save(noise_image_path, "JPEG")
        init_image = load_image(noise_image_path, transform).to(device)  # 加载并预处理
    generated_img = init_image.clone().requires_grad_(True)  # 启用梯度

    vgg_model = VGG(content_layers, style_layers).to(device).eval()  # 实例化模型和优化器
    # print(vgg_model.model)  # 打印vgg模型结构
    optimizer = optim.Adam([generated_img], lr=args.learning_rate)  # 直接对图像本身进行优化

    content_features, _ = vgg_model(content_img)  # 计算内容图的内容特征
    _, style_features = vgg_model(style_img)  # 计算风格图的风格特征

    for epoch in range(args.epochs):

        style_running = 0.0
        content_running = 0.0
        total_running = 0.0
        min = 100

        p_bar = tqdm(range(args.steps_per_epoch), desc=f'epoch {epoch}')
        for step in p_bar:
            generated_content, generated_style = vgg_model(generated_img)  # 计算生成图片的不同层次的内容特征和风格特征
            # 不同层次的内容和风格特征损失加权求和
            content_loss = sum(
                args.content_weight * content_layers[name] * calculate_content_loss(content_features[name], gen_content)
                for name, gen_content in generated_content.items())
            style_loss = sum(
                args.style_weight * style_layers[name] * calculate_style_loss(style_features[name], gen_style) for
                name, gen_style in generated_style.items())

            total_loss = style_loss + content_loss
            if total_loss < min:
                min = total_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generated_img, max_norm=1)  # 梯度裁剪
            optimizer.step()

            # 记录当前step的loss值
            style_loss_item = style_loss.item()
            content_loss_item = content_loss.item()
            total_loss_item = total_loss.item()

            style_running += style_loss.item()
            content_running += content_loss.item()
            total_running += total_loss.item()

            style_losses.append(style_loss_item)
            content_losses.append(content_loss_item)
            total_losses.append(total_loss_item)

            p_bar.set_postfix(style_loss=style_loss.item(), content_loss=content_loss.item())

        # 保存生成图像
        if total_loss == min:
            save_image1(generated_img, args.output_dir, f'epoch{epoch + 1}.jpg')

        # 计算平均值
        avg_style = style_running / args.steps_per_epoch
        avg_content = content_running / args.steps_per_epoch
        avg_total = total_running / args.steps_per_epoch

        # 写入记录
        with open(filepath, "a", newline="") as f:
            m_writer = csv.writer(f)
            m_writer.writerow([epoch, avg_style, avg_content, avg_total])

        # 打印epoch摘要
        # print("\n")
        # print(f"Epoch {epoch} Summary:")
        # print(f"Style Loss: {avg_style:.4f}")
        # print(f"Content Loss: {avg_content:.4f}")
        # print(f"Total Loss: {avg_total:.4f}\n")


def draw():
    plt.figure(figsize=(10, 6))
    plt.plot(total_losses, label='Total Loss', linewidth=1)
    plt.plot(style_losses, label='Style Loss', linewidth=1)
    plt.plot(content_losses, label='Content Loss', linewidth=1)
    plt.xlabel('Training Step')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.title('Loss Curve during Training')
    plt.grid(True)
    plt.savefig(args.output_dir + '/training_loss_curve.png')  # 保存为图片
    plt.close()  # 关闭图像


if __name__ == "__main__":
    run()
    draw()
