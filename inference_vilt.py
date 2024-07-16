import re
import random
import os
import time
import shutil
import argparse
import yaml
from glob import glob
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
from transformers import ViltProcessor
from transformers import ViltForQuestionAnswering

from src.utils import set_seed, VQA_criterion
from src.datasets import VQADataset_vilt
from src.models.vilt_model import VQAModel_vilt

answer_dict_path = "/data/daichi/VQA/class_mapping.csv"
model_dir = "/home/daichi/dlb/dl_lecture_competition_pub/output/2024-07-12/09-55-38/models"
# model_path = "/home/daichi/dlb/dl_lecture_competition_pub/output/2024-07-11/17-06-30/models/model_25.pth"
model_patha = os.path.join(model_dir, os.listdir(model_dir)[0])
# save_path = "/".join(model_path.split("/")[:-1])
# csv_name = "submission_25_change_transform.npy"


def initialize_processor():
    return ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")


def set_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_datasets(data_dir, processor, answer_dict_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset_vilt(df_path=data_dir/"train.json", image_dir=data_dir/"train",
                                    processor=processor, answer_dict_path=answer_dict_path, transform=transform)
    test_dataset = VQADataset_vilt(df_path=data_dir/"valid.json", image_dir=data_dir/"valid",
                                   processor=processor, answer_dict_path=answer_dict_path, transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)
    return train_dataset, test_dataset


def load_model(train_dataset, model_path, device):
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                     id2label=train_dataset.answer2idx,
                                                     label2id=train_dataset.idx2answer)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model


def perform_inference(model, test_loader, device, train_dataset):
    model.eval()
    submission = []
    for batch in tqdm(test_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(**batch)
        # pred = pred.argmax(1).cpu().item()
        logits = pred.logits
        predicted_classes = torch.sigmoid(logits)
        pred = torch.argmax(predicted_classes, dim=1).cpu().item()
        submission.append(pred)
    submission = [train_dataset.idx2answer[id] for id in submission]
    return np.array(submission)


def save_submission(submission, save_path, csv_name):
    np.save(os.path.join(save_path, csv_name), submission)


def main():
    print("======= process start =======")

    processor = initialize_processor()
    set_seed(42)
    device = set_device()

    data_dir = Path("/data/daichi/VQA")
    answer_dict_path = "/data/daichi/VQA/class_mapping.csv"
    model_dir = "/home/daichi/dlb/dl_lecture_competition_pub/output/2024-07-12/09-55-38/models"
    model_paths = glob(os.path.join(model_dir, "*.pth"))

    train_dataset, test_dataset = create_datasets(
        data_dir, processor, answer_dict_path)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False)

    for model_path in model_paths:
        model = load_model(train_dataset, model_path, device)
        print("======= inference start =======")
        submission = perform_inference(model, test_loader, device, train_dataset)
        print("======= inference done =======")
        save_path = "/".join(model_path.split("/")[:-1])
        csv_name = f'submission_{model_path.split("_")[-1].split(".")[0]}.npy'
        save_submission(submission, save_path, csv_name)

    print("======= process done =======")


if __name__ == "__main__":
    main()




# print("======= process start =======")

# processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# set_seed(42)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # dataloader / model
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     # transforms.RandomHorizontalFlip(),
#     transforms.ToTensor()
# ])
# data_dir = Path("/data/daichi/VQA")
# train_dataset = VQADataset_vilt(df_path=data_dir/"train.json", image_dir=data_dir/"train",
#                                 processor=processor, answer_dict_path=answer_dict_path, transform=transform)
# test_dataset = VQADataset_vilt(df_path=data_dir/"valid.json", image_dir=data_dir/"valid",
#                                 processor=processor, answer_dict_path=answer_dict_path, transform=transform, answer=False)
# test_dataset.update_dict(train_dataset)

# test_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=1, shuffle=False)
# # train, validで最も長いテキストは56単語だったため
# # vocab_size = tokenizer.vocab_size
# # model = VQAModel(vocab_size=vocab_size, n_answer=len(
# #     train_dataset.answer2idx)).to(device)
# model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
#                                                  id2label=train_dataset.answer2idx,
#                                                  label2id=train_dataset.idx2answer)
# model.load_state_dict(torch.load(model_path))

# print("======= inference start =======")

# model.eval()
# submission = []
# for image, question in tqdm(test_loader):
#     image, question = image.to(device), question.to(device)
#     pred = model(image, question)
#     pred = pred.argmax(1).cpu().item()
#     submission.append(pred)

# print("======= inference done =======")

# submission = [train_dataset.idx2answer[id] for id in submission]
# submission = np.array(submission)
# # torch.save(model.state_dict(), "model.pth")
# np.save(os.path.join(save_path, csv_name), submission)

# print("======= process done =======")
