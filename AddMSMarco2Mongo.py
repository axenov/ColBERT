#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:08:16 2020

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
        
def create_triple(current_qid, _tmp_results):
    """
    Creates triple in the same format as the original MSMarco Dataset
    
    """
    try: 
        pos_res = find_positive(str(current_qid))
        pos_doc = find_doc(pos_res['doc_id'])
    
        if _tmp_results[0]['doc_id'] != pos_res['doc_id']: 
            neg_res = _tmp_results[0]     
        else:
            neg_res = _tmp_results[1] 
        
        #Positive result
        
        #Combine to query, pos, neg 
        tuple_url = [pos_res['query'], pos_res['doc_url'].strip(), neg_res['url']]
        tuple_title = [pos_res['query'], pos_doc['title'], neg_res['title']] 
        return tuple_url, tuple_title
    
    except: 
        #pdb.set_trace() 
        return None
        

if __name__ == '__main__':  
    client = MongoClient('mongodb://localhost:27017/')
    db = client.msmarco
    table = db.documents
    
    """
    Add documents to mongodb for a fast look up of DID -> url/title/content
    """
    
    if False:
        path = Path('/media/data/05_Download_Databases/71_MSMARCO/documents/msmarco-docs.tsv.gz')
        with gzip.open(path,'rt') as f:
            _tmp_lines = []
            for index, line in tqdm(enumerate(f)):
                #print(line)
                doc_id, url, title, content = line.split('\t')
                line_dict = {'doc_id': doc_id, 
                             'url': url, 
                             'title': title}
                             #'content': content}
                
                _tmp_lines.append(line_dict)
                if index % 10000 == 0 and index > 0: 
                    table.insert_many(_tmp_lines)
                    _tmp_lines = []
                #pdb.set_trace()
                
        resp = table.create_index([ ("doc_id", -1) ])
    
    
    
    """
    Read positive examples: query: DID, Link
    
    """
    table_orcas = db.orcas
    if False:
        path = Path('/media/data/05_Download_Databases/71_MSMARCO/ORCAS/orcas.tsv.gz')
        with gzip.open(path,'rt') as f:
            _tmp_lines = []
            for index, line in tqdm(enumerate(f)):
                #print(line)
                qid, query, doc_id, doc_url = line.split('\t')
                
                line_dict = {'qid': qid, 
                             'query': query, 
                             'doc_id': doc_id,
                             'doc_url': doc_url}
                
                _tmp_lines.append(line_dict)
                if index % 5000 == 0 and index > 0: 
                    table_orcas.insert_many(_tmp_lines)
                    _tmp_lines = []
                    
        resp = table_orcas.create_index([ ("qid", -1) ])
            #pdb.set_trace()
            
    
    
    
    path = Path('/media/data/05_Download_Databases/71_MSMARCO/ORCAS/orcas-doctrain-top100.gz')
    path_txt = Path('/media/philipp/F25225165224E0D96/tmp/Mongodb/orcas-doctrain-top100')
    if False: 
        """
        Rewrite file so that only rank 1 and 2 stay
        """
        ranks_distilled = open('/media/philipp/F25225165224E0D96/tmp/Mongodb/orcas-doctrain-top2.txt', 'w')
        #with gzip.open(path,'rt') as f:
        with open(path_txt, 'r') as f:
            #for index, line in tqdm(enumerate(f)):
            for line in f:
                #print(line)
                qid, _, doc_id, rank, score, _ = line.split(' ')   
                
                #When next series started compute everything from the 100 before
                if int(rank) == 1 or int(rank) == 2:
                    ranks_distilled.write(line)
        ranks_distilled.close()
        
        
    
    #doc_id = "D1555982"       
    #qid = '2000000'
    """
    Create negative samples
    """
    #triples_urls = []
    #triples_titles = []
    outfile_urls = open('orcas_urls.tsv', 'w')
    outfile_titles = open('orcas_titles.tsv', 'w')
    
    _tmp_results = []
    counter = 0
    
    with open('/media/philipp/F25225165224E0D96/tmp/Mongodb/orcas-doctrain-top2.txt', 'r') as f:
    #with gzip.open(path,'rt') as f:
        _tmp_lines = []
        for index, line in tqdm(enumerate(f)):
            #print(line)
            qid, _, doc_id, rank, score, _ = line.split(' ')   
            
            #When next series started compute everything from the 100 before
            if int(rank) == 1:
                if index > 1:
                    #print(_tmp_results)
                    try: 
                        tuple_url, tuple_title = create_triple(current_qid, _tmp_results)
                        outfile_urls.write("\t".join(tuple_url) + '\n')
                        outfile_titles.write("\t".join(tuple_title) + '\n')
                    except TypeError:
                        counter += 1
                        #print(counter)
                    #triples_urls.append(tuple_url)
                    #triples_titles.append(tuple_title)
                    #pdb.set_trace()
                
                current_qid = copy.copy(qid)
                _tmp_results = []
            
                
            elif int(rank) == 2:   
                doc = find_doc(doc_id)
                #if doc: 
                    #print(doc['url'], doc['title'])
                
                if current_qid == qid: 
                    _tmp_results.append(doc)
        
            else: 
                None
                #Dont do anything from rank 3-100
            #if index == 100000: 
            #    break
            
    outfile_urls.close()
    outfile_titles.close()
            
            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            