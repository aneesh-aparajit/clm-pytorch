# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import logging

from datasets import DatasetDict, load_dataset
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, GPT2TokenizerFast,
                          Trainer, TrainingArguments)

from config import CFG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
#                                   Datasets                                   #
# ---------------------------------------------------------------------------- #
def get_datasets():
    ds_train = load_dataset(
        'oscar', 'unshuffled_deduplicated_ta', split="train")
    ds_valid = load_dataset("csv", data_files="../data/ta_valid.csv")['train']
    raw_ds = DatasetDict({
        'train': ds_train,
        'valid': ds_valid
    })
    print(raw_ds)
    return raw_ds


# ---------------------------------------------------------------------------- #
#                             Initialize Tokenizers                            #
# ---------------------------------------------------------------------------- #
tokenizer = GPT2TokenizerFast(
    tokenizer_file='../ckpts/tamil/tokenizer/tokenizer.json')
tokenizer.add_special_tokens({
    'bos_token': '<s>',
    'eos_token': '</s>',
    'unk_token': '<unk>',
    'pad_token': '<pad>'
})

# ---------------------------------------------------------------------------- #
#                               Intialize Config                               #
# ---------------------------------------------------------------------------- #
config = AutoConfig.from_pretrained(
    "abinayam/gpt-2-tamil",
    vocab_size=len(tokenizer),
    n_ctx=CFG.context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# ---------------------------------------------------------------------------- #
#                              Initialize CausalLM                             #
# ---------------------------------------------------------------------------- #
model = AutoModelForCausalLM.from_config(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

# ---------------------------------------------------------------------------- #
#                                Define Collator                               #
# ---------------------------------------------------------------------------- #


def tokenize(item):
    outputs = tokenizer(
        text=item['text'],
        truncation=True,
        max_length=CFG.context_length,
        return_overflowing_tokens=True,
        padding=True,
        return_length=True,
        return_tensors="pt"
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == CFG.context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ---------------------------------------------------------------------------- #
#                             Define Training Args                             #
# ---------------------------------------------------------------------------- #
args = TrainingArguments(
    output_dir="../ckpts/tamil/clm/",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    evaluation_strategy="steps",
    eval_steps=2500,
    logging_steps=500,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    weight_decay=0.01,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=3e-5,
    save_steps=2500,
    fp16=True,
    push_to_hub=True,
)


# ---------------------------------------------------------------------------- #
#                                 Main Function                                #
# ---------------------------------------------------------------------------- #
def main():
    raw_dataset = get_datasets()
    tokenized_datasets = raw_dataset.map(
        tokenize, batched=True, remove_columns=raw_dataset["train"].column_names
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )
    trainer.train()


if __name__ == '__main__':
    main()
