import pandas as pd
from collections import Counter
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,TensorDataset, DataLoader, random_split
from transformers import BertTokenizer, RobertaTokenizer
from transformers import BertForSequenceClassification, AdamW, RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup
from transformers import PreTrainedModel
from transformers import AutoModel




# The Multi-head Attention layer 

class MultiheadAttention_test(nn.Module):
    # n_heads：number of heads
    # hid_dim：the hidden dimension
    def __init__(self, hid_dim, n_heads, dropout):
        super(MultiheadAttention_test, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)# dimension reduction
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to('cuda')

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, (self.hid_dim) //
                   self.n_heads).permute(0, 2, 1, 3)

        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        # attention：[64,6,12,10]
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # x: [64,6,12,50]
        x = torch.matmul(attention, V)


        # x: [64,6,12,50] -> [64,12,6,50]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x: [64,12,6,50] -> [64,12,300]
        x = x.view(bsz, -1, self.n_heads * ((self.hid_dim) // self.n_heads))
        x = self.fc(x)
        return x


# Attention Concatenation Test


class Combo_Model_Attention(nn.Module):
    def __init__(self):
        super(Combo_Model, self).__init__()
        
        self.h_model = AutoModel.from_pretrained('/trained_models/h_pre')
        self.s_model = AutoModel.from_pretrained('/trained_models/s_pre')
        self.base_model = AutoModel.from_pretrained('vinai/bertweet-base')
        self.dropout = nn.Dropout(0.1)
        self.layernorm = nn.LayerNorm(768, eps=1e-05, elementwise_affine=True)
        self.linear1 = nn.Linear(768, 2, bias = True) # output features from bert is 768 and 2 is ur number of labels
        self.linear2 = nn.Linear(768, 768, bias = True)
        self.attention = MultiheadAttention_test(768, 6, dropout=0.1)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        #self.attention.to('cuda')
        output_s = self.h_model(input_ids, attention_mask)[0]
        output_h = self.s_model(input_ids, attention_mask)[0]
        output_base = self.base_model(input_ids, attention_mask)[0]
        
        # You write you new head here
        #output_x[:,0,:], batch_size*768
        outputs_combine = torch.cat((output_base[:,0,:].view(-1,1,768),output_s[:,0,:].view(-1,1,768),output_h[:,0,:].view(-1,1,768)),1)#.to('cuda')
        # The size of outputs :[BatchSize*3*768]
        output_afterAttention = self.attention(outputs_combine, outputs_combine, outputs_combine)[:,0,:]
        # The size of outputs :[BatchSize*1*768]
        #outputs_final = self.dropout(output_afterAttention.view(-1,768))
        outputs_final = output_afterAttention.view(-1,768)
        # Some candidate layers,  may impact the performance accordingly
        #outputs_final = self.layernorm(outputs_final)
        #outputs_final = self.linear2(outputs_final)
        #outputs_final = self.dropout(outputs_final)
        logits = self.linear1(outputs_final)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            labels = labels.float()
            labels_h= torch.eye(2)[labels.long(), :].to('cuda')
            
            loss = loss_fct(
                logits.view(-1, 2), labels_h.view(-1, 2)
            )
            return [loss,logits]
        else:
            return [logits]


#Average, Max polling and concatenation (sequentially) test
class Combo_Model_Other(nn.Module):
    def __init__(self):
        super(Combo_Model, self).__init__()
        
        self.h_model = AutoModel.from_pretrained('/content/drive/MyDrive/h_pre')
        self.s_model = AutoModel.from_pretrained('/content/drive/MyDrive/s_pre') 
        self.base_model = AutoModel.from_pretrained('vinai/bertweet-base')                                        
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(768, 2, bias = True) # output features from bert is 768 and 2 is ur number of labels
        #self.linear = nn.Linear(1536, 2) # output features from bert is 768 and 2 is ur number of labels
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        output_s = self.h_model(input_ids, attention_mask)[0]
        output_h = self.s_model(input_ids, attention_mask)[0]
        output_base = self.base_model(input_ids, attention_mask)[0]
        # to configure different concatenation approach here
        #outputs = (output_s[:,0,:]+output_h[:,0,:])/2
        outputs = torch.max(output_s[:,0,:],output_h[:,0,:])
        #print (outputs.size())
        outputs = torch.max(outputs,output_base[:,0,:])
        #outputs = torch.torch.cat((output_s[:,0,:],output_h[:,0,:]),1)

        sequence_output = self.dropout(outputs)
        logits = self.linear(sequence_output)
        #outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            labels = labels.float()
            labels_h= torch.eye(2)[labels.long(), :].to('cuda')
            
            loss = loss_fct(
                logits.view(-1, 2), labels_h.view(-1, 2)
            )
            #outputs = (loss,) + outputs
            return [loss,logits]
        else:
            return [logits]

