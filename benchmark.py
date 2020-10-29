import torch
from src.model import ColBERT
from src.utils import print_message, load_checkpoint
import torch
query_maxlen = 32
doc_maxlen = 180
dim = 128
similarity = 'cosine'
colbert = ColBERT.from_pretrained('bert-base-multilingual-uncased', 
                                  query_maxlen=query_maxlen, 
                                  doc_maxlen=doc_maxlen, 
                                  dim=dim, 
                                  similarity_metric=similarity)
DEVICE = torch.device("cuda:0")
colbert = colbert.to(DEVICE)
checkpoint_path = 'colbert-200000.dnn'
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
colbert.load_state_dict(checkpoint['model_state_dict'])#Testing inference time (only encoding time!!!)
sen = """Wie viel Grubenwasser wurde aus der Zeche Walsum im Jahr 2017 abgepumpt. 
Wie viel Grubenwasser wurde aus der Zeche Walsum im Jahr 2017 abgepumpt. 
Wie viel Grubenwasser wurde aus der Zeche Walsum im Jahr 2017 abgepumpt."""

sens = [sen] * 20
sens = [sen + ' ' + str(idx) for idx,sen in enumerate(sens)]
import time
for i in range(10):
	start_time = time.time()
	D = colbert.doc(sens)
	end_time = time.time()
	duration = end_time-start_time
	print(duration)
