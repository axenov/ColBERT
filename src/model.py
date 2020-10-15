import string
import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, BertTokenizer
from src.parameters import DEVICE
import pdb

class ColBERT(BertPreTrainedModel):
    def __init__(self, config, query_maxlen, doc_maxlen, dim=128, similarity_metric='cosine'):
        super(ColBERT, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.skiplist = {w: True for w in string.punctuation}

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)

        self.init_weights()
        
        
        """
        Extended by Phil
        """
        
        a = torch.tensor([  101,     1, 11720, 10125, 10103, 41464, 62513, 10163, 10151, 10103,
        12501, 10652, 10108, 10873, 58891,   118, 10932, 41651, 13101, 10110,
        10104, 21531, 10171, 10103, 72010, 15018, 11763,   117, 10103, 41342,
        12748, 17611, 10114, 14709,   143, 32279, 17028, 37182, 10114, 10103,
        10619, 49425, 41464,   119,   102], device='cuda:0')
        """
        SET HERE 
        """
        self.input_dim_manual = 1000
        self.encoded_sentence = a.expand(self.input_dim_manual, 45)
        
        self.attention_mask = torch.ones(self.input_dim_manual, 45, device='cuda:0')
        

    def forward(self, Q, D):
        return self.score(self.query(Q), self.doc(D))

    def query(self, queries):
        queries = [["[unused0]"] + self._tokenize(q) for q in queries]

        input_ids, attention_mask = zip(*[self._encode(x, self.query_maxlen) for x in queries])
        input_ids, attention_mask = self._tensorize(input_ids), self._tensorize(attention_mask)

        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, docs, return_mask=False):
        """
        docs = [["[unused1]"] + self._tokenize(d)[:self.doc_maxlen-3] for d in docs]

        lengths = [len(d)+2 for d in docs]  # +2 for [CLS], [SEP]
        d_max_length = max(lengths)

        input_ids, attention_mask = zip(*[self._encode(x, d_max_length) for x in docs])
        input_ids, attention_mask = self._tensorize(input_ids), self._tensorize(attention_mask)
        pdb.set_trace()
        """
        
        input_ids =  self.encoded_sentence
        attention_mask = self.attention_mask
        
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        """
        # [CLS] .. d ... [SEP] [PAD] ... [PAD]
        mask = [[1] + [x not in self.skiplist for x in d] + [1] + [0] * (d_max_length - length)
                for d, length in zip(docs, lengths)]
        """
        mask_single =[1, True, True, True, True, True, True, True, True, True, True, True, True, 
                      True, True, False, True, True, True, True, True, True, True, True, True, True, 
                      True, False, True, True, True, True, True, True, True, True, True, True, True, 
                      True, True, True, True, False, 1]
        mask = [mask_single] * self.input_dim_manual
        D = D * torch.tensor(mask, device=DEVICE, dtype=torch.float32).unsqueeze(2)
        D = torch.nn.functional.normalize(D, p=2, dim=2)

        return (D, mask) if return_mask else D

    def score(self, Q, D):
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)
        
        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def _tokenize(self, text):
        if type(text) == list:
            return text

        return self.tokenizer.tokenize(text)

    def _encode(self, x, max_length):
        input_ids = self.tokenizer.encode(x, add_special_tokens=True, max_length=max_length)

        padding_length = max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + [103] * padding_length

        return input_ids, attention_mask

    def _tensorize(self, l):
        return torch.tensor(l, dtype=torch.long, device=DEVICE)
