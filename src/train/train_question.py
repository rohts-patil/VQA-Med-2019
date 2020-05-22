_author_ = "Rohit Patil"

import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW

from src.data.make_dataset import (
    get_text_data_train,
    get_text_data_valid,
    SentenceBERTDataset,
)
from src.models.question_model import BertClassifier

MAX_LEN = 128
BATCH_SIZE = 32
BERT_MODEL_NAME = "bert-base-uncased"


def run():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("using ", device)

    train_path = r"train/QAPairsByCategory"
    valid_path = r"val/QAPairsByCategory"

    df_train = get_text_data_train(train_path)
    df_valid = get_text_data_valid(valid_path)

    train_dataset = SentenceBERTDataset(
        question=df_train.question.values,
        target=df_train.label.values,
        max_len=MAX_LEN,
        bert_model_name=BERT_MODEL_NAME,
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(train_dataset)
    )

    valid_dataset = SentenceBERTDataset(
        question=df_valid.question.values,
        target=df_valid.label.values,
        max_len=MAX_LEN,
        bert_model_name=BERT_MODEL_NAME,
    )

    valid_data_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(train_dataset)
    )

    model = BertClassifier(BERT_MODEL_NAME, 4)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-5)
