import logging
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from datasets import DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import (CONFIG_MAPPING, AutoConfig, AutoModelForCausalLM,
                          AutoTokenizer, HfArgumentParser, TrainingArguments)

logger = logging.getLogger(__name__)


def get_datasets():
    ds_train = load_dataset('oscar', 'unshuffled_deduplicated_ta', split="train")
    ds_valid = load_dataset("csv", data_files="../data/ta_valid.csv")['train']
    raw_ds = DatasetDict({
        'train': ds_train, 
        'valid': ds_valid
    })
    print(raw_ds)
    return raw_ds


if __name__ == '__main__':
    raw_dataset = get_datasets()
