#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:21:59 2020

@author: philipp
"""

import gzip
from pathlib import Path
import pdb
import json 
import pickle
from pymongo import MongoClient
from tqdm import tqdm 
from collections import Counter
from typing import Iterable, Iterator, Sequence, Sized, Type
import numpy as np 
import hashlib 
import copy
import requests 
from pathlib import Path 
import sys 

from ORCAS2Passages import SOLR_DEFAULT_URL, CORENAME, getSmallHighlight

"""
Copied from ORCAS2Passages 
"""
def find_doc(doc_id):
    res = table.find({'doc_id' : doc_id})
    a = list(res)
    if len(a) > 0: 
        return a[0]
    else: 
        return None
        
def find_positive(qid): 
    res = table_orcas.find({'qid' : qid})
    a = list(res)
    if len(a) > 0: 
        return a[0]
    else: 
        return None
    

if __name__ == '__main__': 
    client = MongoClient('mongodb://localhost:27017/')
    db = client.msmarco
    table = db.documents_full
    table_orcas = db.orcas
    
    ORCAS_TOP_100 = Path('/media/data/05_Download_Databases/71_MSMARCO/ORCAS/orcas-doctrain-top100')
    
    
    QUESTION_WORDS = ["what", "how", "where","when", "why", "who", "which"]
    
    top100_dev = open('ORCAS_Evaluation/orcas_top100dev.tsv', 'w')
    qrels = open('ORCAS_Evaluation/orcas_qrels.tsv', 'w')
    
    top_100 = []
    
    
    SKIP_LINES = 50000
    with open(ORCAS_TOP_100, 'r') as f:
        for index, line in tqdm(enumerate(f)):
        #for line in f:
            #print(line)
            qid, _, doc_id, rank, score, _ = line.split(' ') 
            if index == 0: 
                qid_old = copy.copy(qid) 
            
            #Every 100 line sprint output
            if qid != qid_old:
                try: 
                    """
                    Only use queries with question word
                    isquestion = [qword in query['query'] for qword in QUESTION_WORDS]
                    if not any(isquestion): 
                        raise TypeError
                    else: 
                        print(query['query'])
                    """
                        
                    """
                    Check for correct length
                    """
                    if len(top_100) < 99: 
                        raise IndexError
                    else: 
                        top_100 = top_100[0:99]
                        
                    passage_id_pos = hash(query['doc_id'] + query['qid'])
                    passage_id_pos += sys.maxsize + 1
                    
                    #Only add positive passage here
                    document_pos = find_doc(query['doc_id'])
                    passage_pos = getSmallHighlight(query['query'], document_pos['content'])
    
                    outline_top_100_pos = query['qid'] + '\t' + str(passage_id_pos) + '\t' + query['query'] + '\t' + passage_pos.replace('\n', ' ') + '\n'
                    top_100.append(outline_top_100_pos)
                    print(len(top_100))
                    top100_dev.writelines(top_100)
                    
                    #Write qrel
                    outline_qrels = query['qid'] + '\t' + '0' + '\t' + str(passage_id_pos) + '\t' + '1' + '\n'
                    qrels.write(outline_qrels)
                
                except (IndexError, TypeError): 
                    print('Error in query Document')
                
                top_100 = []
            
            #Problem: Filter out if pos is in top100
            
            try: 
                document = find_doc(doc_id)
                query = find_positive(qid)
                
                """
                #Comment this in for only using query word questions
                isquestion = [qword in query['query'] for qword in QUESTION_WORDS]
                if not any(isquestion): 
                    raise TypeError
                """
                
                if doc_id != query['doc_id']:
                    passage_neg = getSmallHighlight(query['query'], document['content'])
                    
                    passage_id_neg = hash(doc_id + query['qid'])
                    passage_id_neg += sys.maxsize + 1
                    outline_top_100_neg = query['qid'] + '\t' + str(passage_id_neg) + '\t' + query['query'] + '\t' + passage_neg.replace('\n', ' ') + '\n'
                    top_100.append(outline_top_100_neg)
            except (IndexError, TypeError): 
                #print('Error')
                None
            #pdb.set_trace()
            
            qid_old = copy.copy(qid) 

            if index == (SKIP_LINES * 100): 
                break
        
    top100_dev.close()
    qrels.close()
            
            
            