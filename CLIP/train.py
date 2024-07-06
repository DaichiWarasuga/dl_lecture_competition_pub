import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import shutil
from termcolor import cprint
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import torchvision.models as models

from model import CLIP
from datasets import ThingsMEGDataset
from utils import set_seed

def save_learning_curve(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.xlim(left=0)
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Learning Curve')
    plt.savefig(save_path)
    plt.close()

# @hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args):
    set_seed(args.seed)
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers, "drop_last": True}

    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **loader_args)

    # ------------------
    #       Model
    # ------------------
    model = CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=(3, 4, 6, 3),
        vision_width=64,
        vision_patch_size=16,
        context_length=args.batch_size,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=6,
        input_dim=271*281
    )
    if args.pretrained_model_path:
        model.load_state_dict(torch.load(args.pretrained_model_path))
    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ------------------
    #   Start training
    # ------------------
    train_device = torch.device(args.train_device)
    eval_device = torch.device(args.eval_device)

    criterion = torch.nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        total_train_loss = 0
        model.to(train_device)
        model.train()
        torch.cuda.set_device(train_device)
        torch.cuda.empty_cache()
        
        for brainwaves, images, y, subject_idxs in tqdm(train_loader, desc="Train"):
            images = images.to(train_device)
            brainwaves = brainwaves.view(brainwaves.size(0), -1).to(train_device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits_per_image, logits_per_brainwave = model(images, brainwaves)
                labels = torch.arange(images.size(0)).to(train_device)
                loss_i = criterion(logits_per_image, labels)
                loss_b = criterion(logits_per_brainwave, labels)
                loss = (loss_i + loss_b) / 2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        avg_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_loss)

        total_val_loss = 0
        model.to(eval_device)
        model.eval()
        torch.cuda.set_device(eval_device)
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            for brainwaves, images, y, subject_idxs in tqdm(val_loader, desc="valid"):
                images = images.to(eval_device)
                brainwaves = brainwaves.view(brainwaves.size(0), -1).to(eval_device)
                with torch.cuda.amp.autocast():
                    logits_per_image, logits_per_brainwave = model(images, brainwaves)
                    labels = torch.arange(images.size(0)).to(eval_device)
                    loss_i = criterion(logits_per_image, labels)
                    loss_b = criterion(logits_per_brainwave, labels)
                    val_loss = (loss_i + loss_b) / 2
                total_val_loss += val_loss.item()

        model.to(train_device)
        torch.cuda.set_device(train_device)
        torch.cuda.empty_cache()
        
        current_val_loss = total_val_loss / len(val_loader)
        val_losses.append(current_val_loss)

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            save_path = os.path.join(args.save_dir, "model", f'best_model.pth')
            torch.save(model.state_dict(), save_path)
            
        print(f'Epoch: {epoch + 1}, train_Loss: {avg_loss:.4f}, val_Loss: {current_val_loss:.4f}')

        if (epoch + 1) % 5 == 0:
            model = model.to('cpu')
            save_path = os.path.join(args.save_dir, "model", f'clip_{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)

        save_learning_curve(train_losses, val_losses, os.path.join(args.save_dir, 'learning_curve.png'))

if __name__ == "__main__":
    # confgを記録のためコピー
    date = "2024-07-06-2"
    shutil.copytree("/home/daichi/dlb/dl_lecture_competition_pub/CLIP/configs", f"/home/daichi/dlb/dl_lecture_competition_pub/CLIP/outputs/{date}/config", dirs_exist_ok=True)
    os.makedirs(f"/home/daichi/dlb/dl_lecture_competition_pub/CLIP/outputs/{date}/model", exist_ok=True)
    # コマンドライン引数の解析器を設定
    parser = argparse.ArgumentParser(description='Run the CLIP model training with specified configurations.')
    
    # デフォルトのコマンドライン引数を設定
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the main config file')
    parser.add_argument('--vit_config', type=str, default='configs/ViT.yaml', help='Path to the ViT specific config file')

    # コマンドライン引数を解析
    args = parser.parse_args()

    # config.yaml を読み込む
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        # argparse の既存の属性に設定を追加または上書き
        for key, value in config.items():
            setattr(args, key, value)

    # ViT.yaml を読み込む
    with open(args.vit_config, 'r') as f:
        vit_config = yaml.safe_load(f)
        # ネストされたディクショナリを展開して argparse の属性に追加
        for key, value in vit_config['ViT-L/14-336px'].items():
            setattr(args, key, value)

    run(args)
