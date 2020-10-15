"""
Instruction: 
    1. git clone https://github.com/stanford-futuredata/ColBERT.git
    2. Change in ColBERT Source Code /src/model.py#L17
        To: bert-base-multilingual-uncased
    3. Place this script in the ColBERT directory
    4. Download pretrained Network and replace path at "checkpoint_path"
        https://gigamove.rz.rwth-aachen.de/d/id/aDC7irYU7FetaG
    5. Try with 1000 and 2500 passages (Line 47)
    6. Comment in Line 35 to switch to fp16 and run 1000 and 2500 again

"""


import torch
from src.model import ColBERT
from src.utils import print_message, load_checkpoint
import torch
import time

query_maxlen = 32
doc_maxlen = 512
dim = 128
similarity = 'cosine'

colbert = ColBERT.from_pretrained('bert-base-multilingual-uncased', 
                                  query_maxlen=query_maxlen, 
                                  doc_maxlen=doc_maxlen, 
                                  dim=dim, 
                                  similarity_metric=similarity)


DEVICE = torch.device("cuda:0")
colbert = colbert.to(DEVICE)
#colbert = colbert.half()

checkpoint_path = 'colbert-400000.dnn'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
colbert.load_state_dict(checkpoint['model_state_dict'])



passage = """Based on the regulations mandated by the Federal State of North Rhine-Westphalia and in agreement with the faculties, 
                the Rectorate decided to issue a supplementary regulation to the general examination regulations."""
docs = [passage]*1000
""""
1000 is useless here change parameter in src.model.py Line 36
"""


for i in range(1):
    start_time = time.time()
    with torch.no_grad():
        D = colbert.doc(docs)
    end_time = time.time()
    print(f'Duration: {end_time-start_time} s')


##### Check the number of tokens contained in the passage
passage_tokens = colbert.tokenizer.tokenize(passage)
print(f'Passage contains {len(passage_tokens)} tokens.')


"""
Example Results

Tesla T4
    1000
      Duration: 3.3492774963378906 s
      Passage contains 42 tokens
      12.5k tokens/sec
    1000 - half()
        Duration: 1.3397092819213867 s
        31.3k tokens/sec
    2500
        Duration: 8.899807691574097 s
        Passage contains 42 tokens
        11.8k tokens/sec
    2500 - half()
        Duration: 3.4309325218200684 s
        30.6k tokens/sec  
"""





