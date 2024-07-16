from statistics import mode

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from transformers import BertTokenizer
from transformers import ViltProcessor
from pathlib import Path

try:
    from src.preprocs import process_text, synonym_replacement
except:
    from preprocs import process_text, synonym_replacement
# from preprocs import process_text

def load_dict_from_csv(csv_path):
    """
    CSVファイルから辞書を読み込む

    Parameters
    ----------
    csv_path : str
        CSVファイルのパス

    Returns
    -------
    dict
        CSVから読み込んだ辞書
    """
    df = pd.read_csv(csv_path)
    return {row.iloc[0]: row.iloc[1] for _, row in df.iterrows()}

# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, answer_dict_path=None, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクト
        self.answer_dict_path = answer_dict_path
        self.df = pd.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

        # self.processor = BertTokenizer.from_pretrained('bert-base-uncased')

        if self.answer_dict_path:
            self.answer2idx = load_dict_from_csv(self.answer_dict_path)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}
        else:
            self.answer2idx = {}
            self.idx2answer = {}

    #     # 質問文に含まれる単語を辞書に追加
    #     for question in self.df["question"]:
    #         question = process_text(question)
    #         words = question.split(" ")
    #         for word in words:
    #             if word not in self.question2idx:
    #                 self.question2idx[word] = len(self.question2idx)
    #     self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        # self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        # self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        # question = np.zeros(len(self.idx2question) + 1)  # 未知語用の要素を追加
        # question_words = self.df["question"][idx].split(" ")
        # for word in question_words:
        qustion_text = process_text(self.df["question"][idx])
            # try:
            #     question[self.question2idx[word]] = 1  # one-hot表現に変換
            # except KeyError:
            #     question[-1] = 1  # 未知語

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, qustion_text, answers, int(mode_answer_idx)

        else:
            return image, qustion_text

    def __len__(self):
        return len(self.df)
    
class VQADataset_Synonym(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, answer_dict_path=None, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクト
        self.answer_dict_path = answer_dict_path
        self.df = pd.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

        if self.answer_dict_path:
            self.answer2idx = load_dict_from_csv(self.answer_dict_path)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}
        else:
            self.answer2idx = {}
            self.idx2answer = {}

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        # self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        # self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        question_text = process_text(self.df["question"][idx])
        question_text = synonym_replacement(question_text)

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, question_text, answers, int(mode_answer_idx)

        else:
            return image, question_text

    def __len__(self):
        return len(self.df)

class VQADataset_vilt(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, processor, answer_dict_path=None, transform=None, answer=True):
        self.transform = transform
        self.image_dir = image_dir  # 画像ファイルのディレクト
        self.answer_dict_path = answer_dict_path
        self.df = pd.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer
        self.processor = processor

        if self.answer_dict_path:
            self.answer2idx = load_dict_from_csv(self.answer_dict_path)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}
        else:
            self.answer2idx = {}
            self.idx2answer = {}

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)
        
    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        # self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        # self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")#.convert("RGB")
        image = self.transform(image)
        question = process_text(self.df["question"][idx])

        encoding = self.processor(text=question, images=image, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = encoding.input_ids
        token_type_ids = encoding.token_type_ids
        pixel_values = encoding.pixel_values
        attention_mask = encoding.attention_mask
        pixel_mask = encoding.pixel_mask

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）
            batch = {
                "input_ids": input_ids.squeeze(),
                "token_type_ids": token_type_ids.squeeze(),
                "pixel_values": pixel_values.squeeze(),
                "attention_mask": attention_mask.squeeze(),
                "pixel_mask": pixel_mask.squeeze()
            }
            return batch, torch.tensor(answers, dtype=torch.long), mode_answer_idx

        else:
            batch = {
                "input_ids": input_ids.squeeze(),
                "token_type_ids": token_type_ids.squeeze(),
                "pixel_values": pixel_values.squeeze(),
                "attention_mask": attention_mask.squeeze(),
                "pixel_mask": pixel_mask.squeeze()
            }
            return batch

    def __len__(self):
        return len(self.df)


def collate_fn(batch):
    batch_dicts, answers, mode_answers = zip(*batch)
    batch_combined = {}
    for key in batch_dicts[0].keys():
        batch_combined[key] = torch.stack([d[key] for d in batch_dicts])
    return batch_combined, torch.stack(answers), torch.tensor(mode_answers)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    data_dir = Path("/data/daichi/VQA")
    answer_dict_path="/data/daichi/VQA/class_mapping.csv"
    train_dataset = VQADataset(df_path=data_dir/"train.json", image_dir=data_dir/"train", answer_dict_path=answer_dict_path, transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False)
    # for batch, answers, mode_answer in train_loader:
    #     print(batch)
    #     print(answers)
    #     print(mode_answer)

    #     break
    for image, qustion_text, answers, mode_answer in train_loader:
        print(image)
        print(list(qustion_text))
        break