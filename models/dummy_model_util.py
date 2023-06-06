import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import json

model = SentenceTransformer('all-mpnet-base-v2')

data_directory = "data/"
home_directory = "../"

train_file_path = "data/task_4/task_4_news_media_factuality_train.tsv"
dev_file_path = "data/task_4/task_4_news_media_factuality_dev.tsv"
test_gold_file_path = "task_4_news_media_factuality_test.tsv"

def get_inference_results(sentences, task = "fact"):
    embeddings = torch.from_numpy(model.encode(sentences))
    if task == "fact":
        linear = nn.LazyLinear(2)
        labels = ['Factual', 'Not Factual']
        colname = 'Factuality'
    else:
        linear = nn.LazyLinear(3)
        labels = ['Left', 'Center', 'Right']
        colname = 'Bias'
    scores = nn.Softmax(dim = -1)(linear(embeddings)).T.tolist()
    results = pd.DataFrame({
        colname: labels,
        'Scores': scores
    })
    return results

def get_articles(raw_data, tune = False):
    data = []
    for each_row in raw_data['articles']:
        if tune:
            data.append([each_row['content'], each_row['label']])
        else:
            data.append(each_row['content'])
    return data

def get_split_data(split_data):
    data = []
    for news_src in split_data['json_file_path']:
        raw_data = json.loads(open(os.path.join(home_directory, data_directory, news_src)).read())
        cur_data = get_articles(raw_data)
        data.extend(cur_data)
    return data