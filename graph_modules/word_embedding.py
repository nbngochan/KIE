#!/usr/local/bin/python
# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from transformers import AutoModel, AutoTokenizer
from graph_modules.utils import timing
# from vncorenlp import VnCoreNLP

# change path to ./src
PYTHON_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PYTHON_PATH)
sys.path.insert(0, PYTHON_PATH)


class WordEmbedding(torch.nn.Module):
    def __init__(self, bert_mode="multilingual"):
        super().__init__()
        self.bert_model = bert_mode
        if bert_mode == "multilingual":
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
        elif bert_mode == "phobert":
            self.bert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            self.bert_model = AutoModel.from_pretrained("vinai/phobert-base")

    @timing
    def __call__(self, data):
        if isinstance(data, list):
            return self.extract(data)
        elif isinstance(data, str):
            return self.extract([data])

    @torch.no_grad()
    def extract(self, text_line_list):
        device = next(self.parameters()).device
        padded_sequences = self.bert_tokenizer(text_line_list, padding='longest', return_tensors='pt')
        input_ids = padded_sequences["input_ids"]  # Batch size = number of nodes (text)
        attention_mask = padded_sequences["attention_mask"]
        outputs = self.bert_model(input_ids.to(device), attention_mask=attention_mask.to(device))
        results = outputs[0].to("cpu")  # The last hidden-state is the first element of the output tuple

        # only get meaningful features
        text_line_feature_list = [
            np.array(result[:np.sum(attention_mask[index].numpy())]) for index, result in enumerate(results)
        ]

        return text_line_feature_list
