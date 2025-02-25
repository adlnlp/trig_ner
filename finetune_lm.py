import argparse
import random
import os
import torch
# import config
import math

import utils

import numpy as np

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForWholeWordMask
from transformers import TrainingArguments, Trainer

def load_data():
    with open('./data/{}/train.json'.format(args.dataset), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/{}/dev.json'.format(args.dataset), 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    # with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
    #     test_data = json.load(f)

    dataset = DatasetDict({
        data_split: Dataset.from_dict({
            "sentences": [" ".join(x["sentence"]) for x in data],
            "ner": [x["ner"] for x in data]})
        for data_split, data in zip(["train", "dev"],[train_data, dev_data])
    })

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["sentences", "ner"])

    return tokenized_datasets

def tokenize_function(examples):
  result = tokenizer(examples["sentences"], truncation=True, max_length=512, is_split_into_words=False)
  if tokenizer.is_fast:
      result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
  return result

def seed_torch(seed=3306):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import time
import json
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='./config/cadec_finetuning.json')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--pt_name", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--session_name", type=str, default="TestRun")
    parser.add_argument("--seed", type=int, default=1898)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    # config = config.Config(args)

    seed_torch(args.seed)
    # logger = utils.get_logger(config)
    # logger.info(config)
    # config.logger = logger

    # logger.info("Setup")
    tokenizer = AutoTokenizer.from_pretrained(args.pt_name)
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=0.20)

    # logger.info("Loading Data")
    datasets = load_data()

    training_args = TrainingArguments(
        output_dir="models/%s-finetuned-%s" % (args.pt_name.split("/")[-1], args.dataset),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=True,
        logging_strategy="epoch",
        save_only_model=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        num_train_epochs=args.epochs
    )

    model = AutoModelForMaskedLM.from_pretrained(args.pt_name)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()

    eval_results = trainer.evaluate()
    # logger.info(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    # logger.info(trainer.state.log_history)



