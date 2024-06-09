import torch
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from zeugma.embeddings import EmbeddingTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# import warnings
# warnings.filterwarnings('ignore')

drive_root = '/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project'

columns = ["dataset", "val acc", "val macro f1", "val micro f1", "overall trigger rate", "samples"]
res_df = pd.DataFrame(columns=columns)
backdoor_target_class = 0
sample_size = 2000

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def add_column(data_dict, column_name, data):
    data_dict[column_name]=data
    print(column_name, ": ", data)
    if len(data_dict)==len(columns):
        global res_df
        temp_df = pd.DataFrame([data_dict])  # Convert dictionary to DataFrame
        res_df = pd.concat([res_df, temp_df], ignore_index=True)

ag_data = pd.read_csv(os.path.join(drive_root, "DSC253/ag_data/ag_clean.tsv"), on_bad_lines='skip', sep='\t')
entries = os.listdir(os.path.join(drive_root, "DSC253/ag_data"))
file_list = [entry for entry in entries if os.path.isfile(os.path.join(os.path.join(drive_root, "DSC253/ag_data"), entry)) and entry != "ag_clean.tsv"]
file_list = sorted(file_list)
file_list= [
    "ag_bible_p_0.6.tsv",
    "ag_shakespeare_p_0.0.tsv",
    "ag_shakespeare_p_0.6.tsv",
    "ag_shakespeare_p_0.9.tsv"
]
for file in file_list:
    print("\n"+"-"*20+"\n")
    try:
        if not file.endswith(".tsv"):
            print(f"Passing file: {file}")
            continue
        res_dict = dict()
        add_column(res_dict, "dataset", file)
        ag_bible_data = pd.read_csv(os.path.join(drive_root, "DSC253/ag_data",file), on_bad_lines='skip', sep='\t').dropna()
        ag_bible_data["sentence"] = ag_bible_data["sentence"].astype(str)
        poisoned_ag_bible_data = ag_bible_data.sample(sample_size).copy()
        poisoned_ag_bible_data.label = backdoor_target_class
        combined_ag_data = pd.concat([ag_data, poisoned_ag_bible_data], axis=0).reset_index(drop=True)

        ag_data_train, ag_data_test = train_test_split(combined_ag_data, test_size=0.2, random_state=42)
        # do not over-write the original test data (leave it clean)
        ag_data_val, _ = train_test_split(ag_data_test, test_size=0.5, random_state=42)
        ag_data_train, ag_data_val = ag_data_train.reset_index(drop=True), ag_data_val.reset_index(drop=True)
        X_train, y_train = ag_data_train.sentence, ag_data_train.label
        X_val, y_val = ag_data_val.sentence, ag_data_val.label

        tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
        encoded_X_train = tokenizer(X_train.to_list(), padding='max_length', truncation=True, max_length=64)
        encoded_X_val = tokenizer(X_val.to_list(), padding='max_length', truncation=True, max_length=64)
        label_encoder = LabelEncoder()
        encoded_y_train = label_encoder.fit_transform(y_train)
        encoded_y_val = label_encoder.transform(y_val)

        train_dataset = TextDataset(encoded_X_train, encoded_y_train)
        val_dataset = TextDataset(encoded_X_val, encoded_y_val)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', num_labels=4).to(device)
        training_args = TrainingArguments(
            num_train_epochs=3, 
            per_device_train_batch_size=256,
            per_device_eval_batch_size=256, 
            weight_decay=0.01,
            output_dir='save/',
            save_strategy="no"
        )

        trainer = Trainer(
            model=clf,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        trainer.train()

        pred = trainer.predict(val_dataset)
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=-1)
        accuracy = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average='macro')
        micro_f1 = f1_score(labels, preds, average='micro')
        add_column(res_dict, "val acc", accuracy)
        add_column(res_dict, "val macro f1", macro_f1)
        add_column(res_dict, "val micro f1", micro_f1)
        
        encoded_X_test_poisoned = tokenizer(poisoned_ag_bible_data.sentence[:20].to_list(), padding='max_length', truncation=True, max_length=64)
        sample_preds = trainer.predict(TextDataset(encoded_X_test_poisoned, poisoned_ag_bible_data.label[:20].to_list())).predictions
        add_column(res_dict, "samples", dict(zip(poisoned_ag_bible_data.sentence[:20].to_list(), np.argmax(sample_preds, axis=1).tolist())))

        encoded_X_test_poisoned = tokenizer(ag_bible_data.dropna().sentence.to_list(), padding='max_length', truncation=True, max_length=64)
        preds = trainer.predict(TextDataset(encoded_X_test_poisoned, ag_bible_data.dropna().label.astype(int).to_list())).predictions
        overall = np.sum(np.argmax(preds, axis=1)==backdoor_target_class)/len(ag_bible_data.dropna())
        add_column(res_dict, "overall trigger rate", overall)

        res_df.to_csv(os.path.join(drive_root, "DSC253/ag_data/result.csv"), index=False)
        print(f"\nFinished file: {file}")
    except Exception as e:
        print(f"\nProblem file: {file}")
    