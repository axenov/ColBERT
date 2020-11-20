#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:21:50 2020

@author: philipp
"""

from pymongo import MongoClient
import pdb 
import json 
from tqdm import tqdm 

if __name__ == '__main__': 
    #Init Mongo Table
    client = MongoClient('mongodb://localhost:27017/')
    db = client.msmarco
    table = db.orcas


    #res = list(table.find({'query': {'$regex': '.*german.*'}}))
    res = list(table.aggregate(
            [ { '$sample': { 'size': 500000 } } ]
            )
    )
    #res = list(table.find({},{"doc_id": 1})) #Procudes OOM
        
    doc_ids = [part['doc_id'] for part in res]
    print(len(doc_ids))
    doc_ids = list(set(doc_ids))
    print(len(doc_ids))
    
    out_dict = {}
    for doc_id in tqdm(doc_ids): 
        
        links = list(table.find({'doc_id': doc_id}))
        
        #pdb.set_trace()
        
        synonyms = []
        for query in links: 
            synonyms.append(query['query'])
        #print(synonyms)
        
        out_dict[doc_id] = list(set(synonyms))
        
    with open('snyonyms.json', 'w') as outfile: 
        outfile.write(json.dumps(out_dict, indent=1))
            
            
            