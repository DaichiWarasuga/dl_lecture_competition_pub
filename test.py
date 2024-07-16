import torch
import torch.nn as nn
from transformers import BertTokenizer

# ViltProcessorの初期化
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 質問をトークナイズ
question_text = "What is the color of the cat?"
encoding = tokenizer(question_text, padding="max_length", max_length=56, truncation=True, return_tensors="pt")

# BatchEncodingから入力IDを抽出
input_ids = encoding['input_ids']

# インデックスの範囲を確認
max_index = input_ids.max().item()
min_index = input_ids.min().item()
vocab_size = tokenizer.vocab_size

print(f"Max input ID: {max_index}, Min input ID: {min_index}, Vocab size: {vocab_size}")

# 埋め込みレイヤーの初期化
embedding_dim = 768  # 埋め込み次元を仮定
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# CPUで実行
embedding_layer = embedding_layer.cpu()
input_ids = input_ids.cpu()

# Embeddingレイヤーに入力IDを渡す
embeddings = embedding_layer(input_ids)
print(embeddings.size())
