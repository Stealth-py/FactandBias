import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from roberta.multitask_roberta import MultiTaskRobertaForBiasFactualityCLS
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from datasets import Dataset, DatasetDict
import argparse, random, os, json
from typing import List

def set_random_seed(
        seed: int = 42
    ) -> None:
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', default=42, type=int, help='Pass in a seed value.')
    parser.add_argument('--model_path', default='roberta-base', help='Pass huggingface model path here')
    parser.add_argument('--num_epochs', default=10)
    parser.add_argument('--bz', default=16)
    parser.add_argument('--learning_rate', default=1e-4)
    parser.add_argument('--num_tasks', default=2)
    parser.add_argument('--num_labels_per_tasks', default=[3, 3], type=List[int])
    parser.add_argument('--train_path', default="../new_data/train.csv")
    return parser.parse_args()

def train(
        model,
        optimizer,
        
    ):
    pass

if __name__=="__main__":
    args = parse_args()

    seed = args.seed
    model_path = args.model_path
    num_epochs = args.num_epochs
    bz = args.bz
    lr = args.learning_rate
    num_tasks = args.num_tasks
    num_labels_per_task = args.num_labels_per_tasks

    set_random_seed(seed)

    if num_tasks==1:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        model_path = model_path.strip("cus-")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = MultiTaskRobertaForBiasFactualityCLS.from_pretrained(model_path, num_tasks=num_tasks, num_labels_per_task=num_labels_per_task)