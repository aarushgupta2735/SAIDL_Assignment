import time
from bpetokenizer import BPETokenizer
with open("data/wiki.train.txt") as f: data = f.read()

tokenizer = BPETokenizer()
start = time.time()
print("Starting encode...")
t = tokenizer.encode(data[:100000]) # just 100kb
print(f"100kb Time: {time.time() - start:.2f}s")
