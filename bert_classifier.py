from transformers import AlbertTokenizer, AlbertForSequenceClassification, AlbertConfig, AdamW, DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import torch.nn as nn
import json

from json2html import *
import torch
import argparse
import requests
import datetime
import os


def clean_n_split_data(data):
    """
    This method splits the dataset into two chungs, rows with full data (for training) and rows with missing data (for after-training prediction)
    Each row in data dict contains: id, title, content, publish_date, meta_description, sentiment, label
    :param data:
    :return:
    """
    full_data_rows = []
    missing_data_rows = []
    for i in range(len(data)):
        title = data[i]["title"]
        cont = data[i]["content"]
        meta_des = data[i]["meta_description"]
        sent = data[i]["sentiment"]
        label = data[i]["label"]

        if label == "no": label = 0
        if label == "yes": label = 1

        if None not in (title, cont, meta_des, sent, label) and "" not in (title, cont, meta_des, sent, label):

            full_data_rows.append({
                "title": title,
                "content": cont,
                "meta_des": meta_des,
                "sentiment": str(sent),
                "label": label
            })
        else:
            missing_data_rows.append({
                "title": title,
                "content": cont,
                "meta_des": meta_des,
                "sentiment": str(sent),
                "label": label
            })

    print("Rows without missing features: {}\nRows with missing features: {}".format(len(full_data_rows),
                                                                                     len(missing_data_rows)))

    return full_data_rows, missing_data_rows


def prepate_albert_input_format(train_data, missing_data=False):
    # Note: in this model input strategy, I use the title and meta_data to predict the label
    data = []
    for i, row in enumerate(train_data):
        # full_input_string = row["title"] + "[SEP]" + \
        #                     row["meta_des"] + "[SEP]" + \
        #                     row["content"] + "[SEP]" + \
        # str(row["sentiment"])

        # Because of lack of positive data samples, I treat "title" and "meta_data"
        # as seperate inputs
        if row["label"] == 1:
            full_input_string = row["title"]

            x = enc.encode_plus(full_input_string, max_length=200, pad_to_max_length=True)

            data.append({
                "input_ids": torch.tensor(x["input_ids"], device=torch.device(device), dtype=torch.long),
                "attention_mask": torch.tensor(x["attention_mask"], device=torch.device(device), dtype=torch.float),
                "label": torch.tensor(int(row["label"]), device=torch.device(device), dtype=torch.long)
            })

        if row["meta_des"] is not "" or row["meta_des"] is not None:
            full_input_string = str(row["meta_des"])

            x = enc.encode_plus(full_input_string, max_length=200, pad_to_max_length=True)

            data.append({
                "input_ids": torch.tensor(x["input_ids"], device=torch.device(device), dtype=torch.long),
                "attention_mask": torch.tensor(x["attention_mask"], device=torch.device(device), dtype=torch.float),
                "label": torch.tensor(int(row["label"]), device=torch.device(device), dtype=torch.long)
            })
    return data


def make_dir(dir_path):
    """
    Makes a directory if already doesn't exist
    :param dir_path: Directory path to be created
    :return: Directory path (str)
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def make_save_dir():
    master_dir = "/content/gdrive/My Drive/Colab Notebooks/binary_text_classifier"
    now = datetime.datetime.now().strftime("%d-%m-%Y@%H'%M")
    log_path = "{}/{}/{}".format(master_dir, model_name, now)
    save_dir = make_dir(log_path)
    print("save_dir: {}".format(save_dir))
    return save_dir


train_model = True
num_train_epochs = 20

train_batch_size = 8
gradient_accumulation_steps = 1
learning_rate = 0.000625
adam_epsilon = 1e-8
warmup_steps = 0
weight_decay = 0.01
max_grad_norm = 1.0

# model_name = "albert-base-v2"
model_name = "distilbert-base-uncased"
# enc = AlbertTokenizer.from_pretrained(model_name)
enc = DistilBertTokenizer.from_pretrained(model_name)

config = DistilBertConfig(
    sinusoidal_pos_embds=True,
)

# model = AlbertForSequenceClassification.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, config=config)

# TensorBoardX
output_dir = make_save_dir()
tb_writer = SummaryWriter(output_dir)

# Download data
url = "https://jha-ds-test.s3-eu-west-1.amazonaws.com/legal/data.json"
r = requests.get(url)
data = r.json()

# Get Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Split Data into samples with missing labels and not
full_data, to_predict = clean_n_split_data(data)
train_data = full_data[:999]
eval_data = full_data[1000:-1]

# Prepare appropiate input format
input_data = prepate_albert_input_format(train_data)
train_data_loader = DataLoader(dataset=input_data,  batch_size=train_batch_size, shuffle=True)

# Prepare optimizer
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimization_steps = ((len(train_data_loader) * num_train_epochs) // \
                      (train_batch_size * gradient_accumulation_steps)) + 1000

# TODO: Could use NVIDIA Apex for lower precision calculations -> less memory usage
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)


if train_model:
  # !tensorboard --logdir="/content/gdrive/My Drive/Colab Notebooks/binary_text_classifier/albert-base-v2/16-02-2020@21'15"
  print("To visualise data using TensorBoardX -> type in console:\ntensorboard --logdir={}".format(output_dir))
  model.to(device)
  model.train()

  for epoch in trange(int(num_train_epochs), desc="Epoch"):

    for step, batch in enumerate(tqdm(train_data_loader, desc="Training")):
      outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["label"]
      )

      loss = outputs[0]

      # Log the loss to TensorBoardX
      global_step = (epoch * len(train_data_loader)) + (step + 1)
      tb_writer.add_scalar('loss', loss.item(), global_step)
      print("{} Step: {}".format(global_step, loss.item()))

      # Normalise the loss (Simulates average of a batch)
      loss = loss / gradient_accumulation_steps
      loss.backward(retain_graph=True)

      if (step + 1) % gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

soft = nn.Softmax(0)
eval_model = True
model.eval()

if eval_model:
    input_eval_data = prepate_albert_input_format(eval_data)
    eval_data_loader = DataLoader(dataset=input_eval_data, batch_size=1, shuffle=False)

    correct_count = 0
    for step, batch in enumerate(tqdm(eval_data_loader, desc="Evaluating")):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # labels=batch["label"]
        )
        # print("{}:{}".format(batch["label"], outputs[0]))
        for i, one in enumerate(outputs[0]):
            act = soft(one)
            highest = max(act)
            index_of_max = act.tolist().index(highest)
            print(index_of_max)

            if index_of_max == batch["label"]:
                correct_count += 1

    print("\nCorrect: {} / {}".format(correct_count, len(train_data_loader)))

to_pred = True
model.eval()
if to_pred:
    input_data = prepate_albert_input_format(to_predict)
    train_data_loader = DataLoader(dataset=input_data, batch_size=1, shuffle=False)

    for step, batch in enumerate(tqdm(train_data_loader, desc="Generating Labels")):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        for i, one in enumerate(outputs[0]):
            act = soft(one)
            highest = max(act)
            index_of_max = act.tolist().index(highest)
            # print(index_of_max)

            # Map the label back to "no" or "yes"
            placeholder = ""
            if index_of_max == 0:
                placeholder = "no"
            else:
                placeholder = "yes"

            to_predict[step]["label"] = placeholder
            # print(to_predict[step]["label"])


# Create a HTML file from the samples with missing labels
data_processed = json.dumps(to_predict)
formatted_table = json2html.convert(json = data_processed)

html_file_name = os.path.join(output_dir, "predictions.html")

with open(html_file_name, 'w') as f:
  f.write(formatted_table)
  f.close()
