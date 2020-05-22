_author_ = "Rohit Patil"

import glob

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

label_dict = {"C4_Abnormality": 3, "C3_Organ": 2, "C2_Plane": 1, "C1_Modality": 0}


def get_text_data_train(path):
    filenames = glob.glob(path + "/*.txt")
    dfs = []
    for filename in filenames:
        df_temp = pd.read_csv(
            filename, delimiter="|", names=["id", "question", "answer"]
        )
        df_temp["label"] = df_temp.apply(
            lambda x: label_dict.get(
                (filename.split("\\")[-1].replace("_train.txt", ""))
            ),
            axis=1,
        )
        dfs.append(df_temp)
    df = pd.concat(dfs, ignore_index=True)
    df.drop(["id", "answer"], axis=1, inplace=True)
    return df


def get_text_data_valid(path):
    dfs = []
    filenames = glob.glob(path + "/*.txt")
    for filename in filenames:
        df_temp = pd.read_csv(
            filename, delimiter="|", names=["id", "question", "answer"]
        )
        df_temp["label"] = df_temp.apply(
            lambda x: label_dict.get((filename.split("\\")[-1].replace("_val.txt", ""))),
            axis=1,
        )
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)
    df.drop(["id", "answer"], axis=1, inplace=True)
    return df


class SentenceBERTDataset(Dataset):
    def __init__(self, question, target, max_len, bert_model_name):
        self.question = question
        self.target = target
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model_name, do_lower_case=True
        )
        self.max_len = max_len

    def __len__(self):
        return len(self.question)

    def __getitem__(self, item):
        quest = str(self.question[item])

        inputs = self.tokenizer.encode_plus(
            quest,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }
