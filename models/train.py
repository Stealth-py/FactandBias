import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from datasets import Dataset, DatasetDict
import argparse, random, os, json

def set_random_seed(seed: int = 42) -> None:
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
    parser.add_argument('--model_path', default='roberta-base', help='Pass model name here')
    parser.add_argument('--num_epochs', default=10)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--learning_rate', default=1e-4)
    return parser.parse_args()