import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from transformers import BertTokenizer, RobertaTokenizer
from transformers import BertForSequenceClassification, AdamW, RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, AdamW, RobertaForSequenceClassification
from transformers import BertTokenizer, RobertaTokenizer
from transformers import AutoModel, AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from Models import Combo_Model_Attention
from Models import Combo_Model_Other
from Data_Process import Process


class model_trainer():

    def run(self):
        os.environ['WANDB_MODE'] = 'offline'
        os.environ["WANDB_DISABLED"] = "true"

        # For copyright reasons, please contact the author of Parody dataset for data access
        Process("raw_data_path", "train_index_path", "validation_index_path", "test_index_path")
        res_data = Process.run()
        train_dataset = res_data[0]  # processed train dataset
        val_dataset = res_data[1]  # processed validation dataset
        test_dataset = res_data[2]  # processed test dataset

        # To keep the reproducity, we set a random seed as below
        torch.manual_seed(100)
        random.seed(100)
        np.random.seed(100)
        torch.cuda.manual_seed_all(100)


        # TODO: Choose A TOKENIZER
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", do_lower_case=True)

        # TODO: CHOOSE A MODEL
        # 1. BERTweet Model
        model = AutoModelForSequenceClassification.from_pretrained('vinai/bertweet-base', num_labels=2, output_attentions=False, output_hidden_states=False)

        # 2. Multi-Semantic-Encoder Model (Attention)
        # model = Combo_Model_Attention()

        # 3. Multi-Semantic-Encoder Model (Other Approach, need manual config)
        # model = Combo_Model_Other()

        # 4. RoBERTA Model
        # model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2, output_attentions=False, output_hidden_states=False)

        # 5. BERT Model
        # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)

        model.to('cuda')
        training_args = TrainingArguments("test_trainer", per_device_train_batch_size=128, per_device_eval_batch_size=128,
                                          evaluation_strategy="epoch", num_train_epochs=2.0)

        # About 22G GPU memories are needed                                 
        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, tokenizer=tokenizer
        )

        # Train
        trainer.train()

        model.eval()
        torch.save(model,"/trained_models/main_model.pt")

        # TODO: Test the model
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        preds = []
        for i, batch in enumerate(test_dataloader):
            with torch.no_grad():
                # outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device))
                outputs = model(batch['input_ids'].to(device), token_type_ids=None,
                                attention_mask=batch['attention_mask'].to(device))
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                preds.append(logits)
        final_preds = np.concatenate(preds, axis=0)
        # print (len(final_preds))
        final_preds = np.argmax(final_preds, axis=1)


        return
