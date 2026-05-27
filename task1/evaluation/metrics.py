from evaluation import *

def perplexity(loss):
    return torch.exp(loss)

def throughput(num_tokens, time_seconds):
    return num_tokens / time_seconds