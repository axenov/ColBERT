#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:56:11 2020

@author: philipp
"""

from transformers import XLMRobertaTokenizer, XLMRobertaModel

"""
See https://github.com/huggingface/transformers/issues/1413

"""


if __name__ == '__main__': 
    model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
    
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    print(len(tokenizer))
    
    #print(tokenizer.convert_ids_to_tokens(list(range(249990, 250000))))
    tokenizer.add_tokens(["[unused1]"])
    print(len(tokenizer))
    #print(tokenizer.cov)
    
    model.resize_token_embeddings(len(tokenizer)) 

    print(model.embeddings.word_embeddings.weight[-1, :])


    """"
    Compare it with BERT tokenizer 
    """
    from transformers import BertPreTrainedModel, BertModel, BertTokenizer
    
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_basic_tokenize=False)
    bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
    bert_tokenizer.encode("[unused1]", add_special_tokens=False)