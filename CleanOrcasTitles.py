#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:41:27 2020

@author: philipp
"""

import pdb
from tqdm import tqdm 

if __name__ == '__main__':
    uncleaned_titles = "orcas_titles.tsv"
    
    cleaned_titles = open("orcas_titles_cleaned.tsv", 'w')
    
    with open(uncleaned_titles, 'r') as file:
        for line in tqdm(file): 
            query, positive, negative = line.split('\t')
            
            #Check if every part is a valid one
            is_length = [True if (len(part) > 2) else False for part in line.strip().split('\t') ]
            if all(is_length):
                cleaned_titles.write(line)
                #pdb.set_trace()
    
    cleaned_titles.close()