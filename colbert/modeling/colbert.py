import string
import torch
import torch.nn as nn

from transformers import DPRQuestionEncoder, DPRContextEncoder, BertPreTrainedModel
from colbert.parameters import DEVICE
class ColBERT():
    def __init__(self, query_maxlen, doc_maxlen):
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen

        self.query_encoder = DPRQuestionEncoder.from_pretrained('gbert-base-germandpr/query_model').eval()
        self.query_encoder.to(DEVICE)
        self.passage_encoder = DPRContextEncoder.from_pretrained('gbert-base-germandpr/passage_model').eval()
        self.passage_encoder.to(DEVICE)
        
        self.dropout = nn.Dropout(0.1).eval()

    def forward(self, Q, D):
        return self.score(self.query(*Q), self.doc(*D))

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        pooled_output = self.query_encoder(input_ids, attention_mask=attention_mask)['pooler_output']
        pooled_output = self.dropout(pooled_output)
        return pooled_output

    def doc(self, input_ids, attention_mask, keep_dims=True):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        pooled_output = self.passage_encoder(input_ids, attention_mask=attention_mask)['pooler_output']
        pooled_output = self.dropout(pooled_output)
        return pooled_output
    
    def score(self, Q, D):
        scores = torch.matmul(Q, torch.transpose(D, 0, 1))
        scores = nn.functional.log_softmax(scores, dim=1)[0]
        return scores

    def from_pretrained(cls, path):
        return ColBERT(32,150)