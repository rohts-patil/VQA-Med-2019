_author_ = "Rohit Patil"

import datetime
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from src.data.make_dataset import (
    get_text_data_train,
    get_text_data_valid,
    SentenceBERTDataset,
)
from src.models.question_model import BertClassifier

MAX_LEN = 128
BATCH_SIZE = 32
BERT_MODEL_NAME = "bert-base-uncased"
EPOCHS = 2
MODEL_PATH = "./model_save/"


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_questions():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("using ", device)

    train_path = r"src/data/train/QAPairsByCategory"
    valid_path = r"src/data/valid/QAPairsByCategory"

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

    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    training_stats = []

    total_t0 = time.time()

    for epoch_i in range(0, EPOCHS):

        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, EPOCHS))
        print("Training...")

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_data_loader):

            if step % 100 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(
                    "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                        step, len(train_data_loader), elapsed
                    )
                )

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            loss, logits = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

        avg_train_loss = total_train_loss / len(train_data_loader)

        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in valid_data_loader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                (loss, logits) = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(valid_data_loader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(valid_data_loader)

        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "Valid. Accur.": avg_val_accuracy,
                "Training Time": training_time,
                "Validation Time": validation_time,
            }
        )

    print("")
    print("Training complete!")

    print(
        "Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0))
    )

    print_training_timings(training_stats)

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    print("Saving model to %s" % MODEL_PATH)

    torch.save(model.state_dict(), MODEL_PATH)


def print_training_timings(training_stats):
    pd.set_option("precision", 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index("epoch")
    print(df_stats.head())


if __name__ == "__main__":
    train_questions()
