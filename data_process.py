import pandas as pd
from collections import Counter
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset,TensorDataset, DataLoader, random_split
from transformers import BertTokenizer, RobertaTokenizer
from transformers import BertForSequenceClassification, AdamW, RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup

class ParodyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class Process():

    def __init__(self, raw_data_path, train_index_path, validation_index_path,test_index_path):
        self.raw_data_path = raw_data_path
        self.train_index_path = train_index_path
        self.validation_index_path = validation_index_path
        self.test_index_path = test_index_path

    def run(self):
        raw_data = pd.read_csv(self.raw_data_path, lineterminator='\n')
        train_index = pd.read_csv(self.train_index_path, lineterminator='\n',
                                  header=None)
        validation_index = pd.read_csv(self.validation_index_path, lineterminator='\n',
                                       header=None)
        test_index = pd.read_csv(self.test_index_path, lineterminator='\n',
                                 header=None)
        data = pd.DataFrame(raw_data, columns=['tweet_id', 'tweet_pp', 'label'])

        train_data = data.join(train_index.set_index(0), on='tweet_id', how='left').dropna()
        train_data = train_data.drop(columns=['tweet_id', 1])

        validation_data = data.join(validation_index.set_index(0), on='tweet_id', how='left').dropna()
        validation_data = validation_data.drop(columns=['tweet_id', 1])

        test_data = data.join(test_index.set_index(0), on='tweet_id', how='left').dropna()
        test_data = test_data.drop(columns=['tweet_id', 1])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_text_values = train_data['tweet_pp'].values
        validation_text_values = validation_data['tweet_pp'].values
        test_text_values = test_data['tweet_pp'].values
        train_labels = train_data['label'].values
        validation_labels = validation_data['label'].values
        test_labels = test_data['label'].values


        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", do_lower_case=True)
        train_encodings = tokenizer(train_text_values.tolist(), max_length=85, truncation=True, padding=True)  # tried with length 52
        val_encodings = tokenizer(validation_text_values.tolist(), max_length=85, truncation=True, padding=True)
        test_encodings = tokenizer(test_text_values.tolist(), max_length=85, truncation=True, padding=True)

        train_dataset = ParodyDataset(train_encodings, train_labels.tolist())
        val_dataset = ParodyDataset(val_encodings, validation_labels.tolist())
        test_dataset = ParodyDataset(test_encodings, test_labels.tolist())

        return (train_dataset,val_dataset,test_dataset)
