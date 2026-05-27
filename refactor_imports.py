import os
import re

task1_dir = "/mnt/c/Users/akgup/Documents/Obsidian Backup/Vault/Files/SAIDL_ASSIGNMENT/task1"

imports_to_add = """import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config.transformer_config import TransformerConfig
from config.train_config import TrainConfig
"""

def clean_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    content = content.replace("from src import *", imports_to_add)
    content = content.replace("from train import *", imports_to_add)
    content = content.replace("from evaluation import *", imports_to_add)
        
    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Refactored {path}")

for root, dirs, files in os.walk(task1_dir):
    for file in files:
        if file.endswith('.py') and file != '__init__.py':
            clean_file(os.path.join(root, file))
