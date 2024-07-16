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
import matplotlib as plt
import torch
import torch.nn as nn
import torchvision
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from transformers import BertTokenizer

from src.utils import set_seed, VQA_criterion
from src.datasets import VQADataset
from src.models.base import VQAModel

answer_dict_path="/data/daichi/VQA/class_mapping.csv"
model_path = "/home/daichi/dlb/dl_lecture_competition_pub/output/2024-07-16/12-51-14/models/model_12.pth"
save_path = "/".join(model_path.split("/")[:-1])
csv_name = "submission_12.npy"

print("======= process start =======")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
# dataloader / model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
data_dir = Path("/data/daichi/VQA")
train_dataset = VQADataset(df_path=data_dir/"train.json", image_dir=data_dir/"train", answer_dict_path=answer_dict_path, transform=transform)
test_dataset = VQADataset(df_path=data_dir/"valid.json", image_dir=data_dir/"valid", answer_dict_path=answer_dict_path, transform=transform, answer=False)
test_dataset.update_dict(train_dataset)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
# train, validで最も長いテキストは56単語だったため
vocab_size = tokenizer.vocab_size
model = VQAModel(vocab_size=vocab_size, n_answer=len(train_dataset.answer2idx)).to(device)

model.load_state_dict(torch.load(model_path))

print("======= inference start =======")

model.eval()
submission = []
for image, question in tqdm(test_loader):
    question = tokenizer.batch_encode_plus(question, padding='max_length', max_length=60, truncation=True, return_tensors='pt')["input_ids"]
    # print(image, question)
    image, question = image.to(device), question.to(device)
    pred = model(image, question)
    pred = pred.argmax(1).cpu().item()
    submission.append(pred)

print("======= inference done =======")

submission = [train_dataset.idx2answer[id] for id in submission]
submission = np.array(submission)
# torch.save(model.state_dict(), "model.pth")
np.save(os.path.join(save_path, csv_name), submission)

print("======= process done =======")