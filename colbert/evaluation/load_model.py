import os
import ujson
import torch
import random

from collections import defaultdict, OrderedDict

from colbert.parameters import DEVICE
from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message, load_checkpoint


def load_model(args, do_print=True):
    colbert = ColBERT(query_maxlen=args.query_maxlen, doc_maxlen=args.doc_maxlen)
    print_message("#> Loading model checkpoint.", condition=do_print)
    return colbert
