from datasets import concatenate_datasets, load_dataset
from tokenizers import ByteLevelBPETokenizer

from config import CFG

# Extracted from https://github.com/AbinayaM02/GPT2-Tamil/blob/main/src/train_tokenizer.py
dataset = load_dataset('oscar', 'unshuffled_deduplicated_ta', split="train")
indic_tamil = load_dataset(
    "text", data_files="../data/data/ta/ta.txt")['train']
dataset = concatenate_datasets([dataset, indic_tamil])


def batch_iterator(batch_size: int = 512):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i+batch_size]['text']


def train_tokenizer():
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        batch_iterator(),
        vocab_size=52_000,
        min_frequency=2,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ],
    )
    tokenizer.save(f"{CFG.ta_tokenizer_ckpt}/")


if __name__ == '__main__':
    train_tokenizer()
