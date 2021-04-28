
import torch
from transformers import DPRQuestionEncoderTokenizerFast
from colbert.modeling.tokenization.utils import _split_into_batches

class QueryTokenizer():
    def __init__(self, query_maxlen):
        self.tok = DPRQuestionEncoderTokenizerFast.from_pretrained('gbert-base-germandpr/query')
        self.query_maxlen = query_maxlen

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))
        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]
        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']
        return ids

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [Q] marker
        obj = self.tok(batch_text, padding='max_length', truncation=True,
                       return_tensors='pt', max_length=self.query_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']

        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches

        return ids, mask
