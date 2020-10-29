#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:44:43 2020

@author: philipp
"""
from transformers import AutoTokenizer, AutoModelForPreTraining


if __name__ == '__main__': 
    tokenizer = AutoTokenizer.from_pretrained("german-nlp-group/electra-base-german-uncased")
    
    outfile_urls = open('output/orcas_urls.tsv', 'r')
    
    for index, line in enumerate(outfile_urls.readlines()): 
        if index > 25:
            query, pos_url, neg_url = line.split('\t')
            
            tokenized = tokenizer.tokenize(pos_url)
            print(pos_url)
            print(" ".join(tokenized), end='\n\n')
