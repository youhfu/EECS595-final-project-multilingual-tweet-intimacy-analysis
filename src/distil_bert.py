# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import re

from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

from transformers import (
    BertTokenizer,
    DistilBertTokenizer,
    DistilBertModel,
    DistilBertForSequenceClassification,
)

from tqdm.auto import tqdm, trange

# import wandb

# wandb.login()

# %%
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


# %%
# filter "http" and "https" from the text
def filter_websites(text):
    pattern = r"(http|https)"
    text = re.sub(pattern, " ", text)
    return text


# train_df["text"] = train_df["text"].apply(filter_websites)
train_df["text"] = train_df["text"].str.lower()

# %%
test_df["text"] = test_df["text"].apply(filter_websites)
test_df["text"] = test_df["text"].str.lower()

# %%
bert_model = "distilbert-base-multilingual-cased"
bert_tokenizer = DistilBertTokenizer.from_pretrained(bert_model)


# %%
def bert_input(df):
    corpus = bert_tokenizer(
        text=df["text"].to_list(),
        add_special_tokens=True,
        padding="max_length",
        truncation="longest_first",
        max_length=168,
        return_attention_mask=True,
    )
    bert_ids = np.array(corpus["input_ids"])
    bert_attention_masks = np.array(corpus["attention_mask"])
    return bert_ids, bert_attention_masks


bert_ids, bert_attention_masks = bert_input(train_df)

# %%
# find the text with only 6 languages in test_df
languages = train_df["language"].unique()
def_df = test_df[test_df["language"].isin(languages)]

# %%
test_bert_ids, test_bert_attention_masks = bert_input(def_df)

# %%
label_scaler = StandardScaler()
label_scaler.fit(train_df["label"].values.reshape(-1, 1))

label_train = label_scaler.transform(train_df["label"].values.reshape(-1, 1))
label_dev = label_scaler.transform(def_df["label"].values.reshape(-1, 1))


# %%
def bertDataLoader(bert_ids, bert_attention_masks, labels, batch_size=32, shuffle=True):
    bert_ids = torch.tensor(bert_ids)
    bert_attention_masks = torch.tensor(bert_attention_masks)
    labels = torch.tensor(labels)
    data = TensorDataset(bert_ids, bert_attention_masks, labels)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# %%
train_dataloader = bertDataLoader(
    bert_ids, bert_attention_masks, label_train, batch_size=4
)
dev_dataloader = bertDataLoader(
    test_bert_ids, test_bert_attention_masks, label_dev, batch_size=4
)


# %%
class bertRegressor(nn.Module):
    def __init__(self, bert_model, dropout=0.1):
        super(bertRegressor, self).__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained(
            bert_model, num_labels=768
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)

    def forward(self, ids, attention_mask):
        pooled_out = self.bert(ids, attention_mask).logits
        drop_output = self.dropout(pooled_out)
        output = self.linear(drop_output)
        return output


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
def evaluate_model(model, dataLoader, device):
    model.eval()
    # evaluate dev data by pearson' r coefficient
    label_list = []
    output_list = []
    with torch.no_grad():
        for ids, attention_mask, label in dataLoader:
            ids = ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)

            test_output = model(ids, attention_mask)
            label_list.append(label)
            output_list.append(test_output)

        label_final = torch.cat(label_list, dim=0)
        output_final = torch.cat(output_list, dim=0)

        label_final = label_final.cpu().detach().numpy().reshape(-1)
        output_final = output_final.cpu().detach().numpy().reshape(-1)

        pearson_corr, _ = pearsonr(label_final, output_final)
    model.train()

    return pearson_corr, output_final


# %%
model = bertRegressor(bert_model)

model.to(device)

learning_rate = 1e-5
epochs = 10
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fcn = nn.MSELoss()


# wandb initialization
# wandb.init(
#     project="intimacy-analysis",
#     config={
#         "batch_size": 32,
#         "embedding_size": 768,
#         "learning_rate": learning_rate,
#         "epochs": epochs,
#     },
# )

loss_100 = 0
model.train()
for epoch in trange(epochs):
    for step, (ids, attention_mask, labels) in enumerate(tqdm(train_dataloader)):
        ids = ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        model.zero_grad()
        output = model(ids, attention_mask)
        loss = loss_fcn(output.view(-1).float(), labels.view(-1).float())
        loss.backward()
        optimizer.step()
        loss_100 += loss.item()
        # if step % 10 == 0:
        #     wandb.log({"loss": loss_100})
        #     loss_100 = 0
    # pearson_corr = evaluate_model(model, dev_dataloader, device)
    # wandb.log({"pearson_corr": pearson_corr})
    # save model after each epoch


# save final model
torch.save(model.state_dict(), "project_model_final.pt")

# split the test data to different languages
test_languages = test_df["language"].unique()
test_data = {}
for lang in test_languages:
    test_data[lang] = test_df[test_df["language"] == lang]

corr_list = []
for key, item in test_data.items():
    test_ids, test_attention_masks = bert_input(item)
    labels = label_scaler.transform(item["label"].values.reshape(-1, 1))
    multi_test_dataloader = bertDataLoader(
        test_ids, test_attention_masks, labels, shuffle=False
    )
    pearson_corr, _ = evaluate_model(model, multi_test_dataloader, device)
    corr_list.append(pearson_corr)

corr_df = pd.DataFrame({"language": test_languages, "pearson_corr": corr_list})
corr_df.to_csv("corr_distil.csv", index=False)

test_ids, test_attention_masks = bert_input(test_df)
labels = label_scaler.transform(test_df["label"].values.reshape(-1, 1))
test_data_loader = bertDataLoader(test_ids, test_attention_masks, labels, shuffle=False)
pearson_corr, pred = evaluate_model(model, test_data_loader, device)
test_df["distil_pred"] = label_scaler.inverse_transform(pred.reshape(-1, 1))
test_df.to_csv("test.csv", index=False)

train_now_dataloader = bertDataLoader(
    bert_ids, bert_attention_masks, label_train, batch_size=4, shuffle=False
)
_, train_pred = evaluate_model(model, train_now_dataloader, device)
train_df["distil_pred"] = label_scaler.inverse_transform(train_pred.reshape(-1, 1))
train_df.to_csv("train.csv", index=False)
