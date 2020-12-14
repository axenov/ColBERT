import os
import time
import faiss
import random
import torch

from colbert.utils.runs import Run
from multiprocessing import Pool
from colbert.modeling.inference import ModelInference
from colbert.evaluation.ranking_logger import RankingLogger

from colbert.utils.utils import print_message, batch
from colbert.ranking.rankers import Ranker

from colbert.evaluation.load_model import load_model_custom

from colbert.parameters import BASE_MODEL

import timeit 

if __name__ == '__main__': 
    #Model Path
    
    checkpoint_path = "/media/data/47_KISS/23_MS_MARCO_BERTS/ColBERTv0.2/experiments/resume_with_german_triples/train.py/2020-12-13_17.05.21/checkpoints/colbert_CP_80k_resumed_german.dnn"
    
    colbert, ckpt = load_model_custom(BASE_MODEL, checkpoint_path, dim=1024)
    
    inference = ModelInference(colbert, amp=False)


    """
    Encoding Section
    """    
    query = ["test"] * 512
    query_encoding = inference.queryFromText(query) 
    
    text = ["hallo test hallo test hallo test hallo test hallo test hallo test hallo test hallo test hallo test"] * 512
    text_encoding = inference.docFromText(text)
    
    
    """
    Benchmark section
    """
    res_text_shape = inference.docFromText(text).shape 
    
    nr_tokens = res_text_shape[0] * res_text_shape[1]
    
    """
    %timeit inference.docFromText(text)
    """
    number_of_runs = 10
    benchmark_time = timeit.timeit('inference.docFromText(text)', globals=globals(), number=number_of_runs)
    
    print('Tokens per second: ', int(nr_tokens * number_of_runs / benchmark_time) )
    
    
    
    
