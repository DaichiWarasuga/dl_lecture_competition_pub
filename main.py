import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from transformers import BertTokenizer

from src.utils import set_seed, VQA_criterion
from src.datasets import VQADataset
from src.models.base import VQAModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 学習の実装
def train(model, dataloader, optimizer, criterion, device):
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

        optimizer.zero_grad()
        # torch.cuda.synchronize()
        loss.backward()
        optimizer.step()

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


def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    data_dir = Path("/data/daichi/VQA")
    train_dataset = VQADataset(df_path=data_dir/"train.json", image_dir=data_dir/"train", transform=transform)
    test_dataset = VQADataset(df_path=data_dir/"valid.json", image_dir=data_dir/"valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    # train, validで最も長いテキストは56単語だったため
    vocab_size = tokenizer.vocab_size
    model = VQAModel(vocab_size=vocab_size, n_answer=len(train_dataset.answer2idx)).to(device)

    # optimizer / criterion
    num_epoch = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # train model
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    # 提出用ファイルの作成
    # model.eval()
    # submission = []
    # for image, question in test_loader:
    #     image, question = image.to(device), question.to(device)
    #     pred = model(image, question)
    #     pred = pred.argmax(1).cpu().item()
    #     submission.append(pred)

    # submission = [train_dataset.idx2answer[id] for id in submission]
    # submission = np.array(submission)
    # torch.save(model.state_dict(), "model.pth")
    # np.save("submission.npy", submission)

if __name__ == "__main__":
    main()
