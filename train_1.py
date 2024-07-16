import re
import random
import os
import time
import shutil
import argparse
import yaml
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from transformers import BertTokenizer

from src.utils import set_seed, VQA_criterion
from src.datasets import VQADataset, VQADataset_Synonym
from src.models.base import VQAModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 学習の実装
def train(model, dataloader, optimizer, criterion, device, scaler):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in tqdm(dataloader):
        # トークナイズ
        question = tokenizer.batch_encode_plus(question, padding='max_length', max_length=60, truncation=True, return_tensors='pt')["input_ids"]
        image = image.to(device)
        question = question.to(device)
        answer = torch.stack(answers).to(device)
        mode_answer = mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in tqdm(dataloader):
        question = tokenizer.batch_encode_plus(question, padding='max_length', max_length=60, truncation=True, return_tensors='pt')["input_ids"]
        image = image.to(device)
        question = question.to(device)
        answer = answers.to(device)
        mode_answer = mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def save_learning_curve(train_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.xlim(left=0)
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Learning Curve')
    plt.savefig(save_path)
    plt.close()

def main(args):
    # deviceの設定
    set_seed(42)
    device = args.device

    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    data_dir = Path("/data/daichi/VQA")
    train_dataset = VQADataset(df_path=data_dir/"train.json", image_dir=data_dir/"train", answer_dict_path=args.answer_dict_path, transform=transform)
    test_dataset = VQADataset(df_path=data_dir/"valid.json", image_dir=data_dir/"valid", answer_dict_path=args.answer_dict_path, transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    # train, validで最も長いテキストは56単語だったため
    vocab_size = tokenizer.vocab_size
    model = VQAModel(vocab_size=vocab_size, n_answer=len(train_dataset.answer2idx)).to(device)

    if args.load_model_path:
        model.load_state_dict(torch.load(args.load_model_path))

    # optimizer / criterion
    num_epoch = args.epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_losses = []
    scaler = torch.cuda.amp.GradScaler()
    # train model
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device, scaler)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")
        train_losses.append(train_loss)
        try:
            save_learning_curve(train_losses, args.save_path)
        except:
            pass
        if (epoch + 1) % 4 == 0:
            model = model.to('cpu')
            torch.save(model.state_dict(), args.model_dir / f"model_{epoch + 1}.pth")
            model = model.to(device)


    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question in test_loader:
        question = tokenizer.batch_encode_plus(question, padding='max_length', max_length=60, truncation=True, return_tensors='pt')["input_ids"]
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    # torch.save(model.state_dict(), "model.pth")
    np.save("submission_last_epoch.npy", submission)

if __name__ == "__main__":
    # 実行結果を保存するフォルダを作成
    # 年月日を取得
    current_time = datetime.now()
    current_time = "/".join(current_time.strftime("%Y-%m-%d %H-%M-%S").split(" "))
    output_dir = Path(f"output/{current_time}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # modelを保存するフォルダを作成
    model_dir = output_dir / "models"
    config_dir = output_dir / "configs"
    model_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "config.yaml"

    # configを保存
    shutil.copytree("/home/daichi/dlb/dl_lecture_competition_pub/configs", config_dir, dirs_exist_ok=True)
    
    # 学習曲線の保存先
    save_path = output_dir / "learning_curve.png"
    # パーサーの設定
    parser = argparse.ArgumentParser()

    # コマンドライン引数を解析
    args = parser.parse_args()

    # config.yaml を読み込む
    with open(os.path.dirname(__file__) / config_path, 'r') as f:
        config = yaml.safe_load(f)
        # argparse の既存の属性に設定を追加または上書き
        for key, value in config.items():
            setattr(args, key, value)
    setattr(args, "model_dir", model_dir)
    setattr(args, "save_path", save_path)

    main(args)
