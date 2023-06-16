from pathlib import Path
from tqdm import tqdm
import json
from datasets import Dataset
import pandas as pd
import os


# Define the function to read a JSON file and return its contents
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file, encoding='unicode_escape')
    return data


# Define the function to load data from a directory into a Dataset object
def load_data_from_directory(directory_path):
    file_paths = sorted(directory_path.glob("*.json"))
    data = list()
    for file_path in tqdm(file_paths):
        if not file_path.name.startswith('._'):
            file_data = read_json_file(file_path)
            data.append(file_data)
    return Dataset.from_list(data)


def iterate_directory(directory_path):
    path = Path(directory_path)

    result = dict()
    result['files'] = dict()
    result['directories'] = dict()
    
    for file in path.iterdir():
        if file.suffix == '.tar.gz' or file.name == '.gitkeep':
            continue  # Ignore .tar.gz files and .gitkeep files
        if file.is_file():
            print(f"Processing file: {file}")
            result['files'][file.name] = file
        elif file.is_dir():
            print(f"Processing directory: {file}")
            # Add your desired operations here for directories
            result['directories'][file.name] = file

    return result


# Below function is for loading task 4 data
def get_articles(raw_data):
    data = []
    for each_row in raw_data['articles']:
        data.append(each_row['content'])
    return data


# Below function is for loading task 4 data
def get_split_data(split_data, data_directory):
    data = []
    labels = []
    for i, news_src in enumerate(split_data['json_file_path']):
        raw_data = json.loads(open(os.path.join(data_directory, news_src)).read())
        cur_data = get_articles(raw_data)
        data.extend(cur_data)
        labels.extend([split_data['label'][i] for _ in range(len(cur_data))])
    final_data = [data, labels]
    return final_data


def create_dataloaders(task_data_main_dir, task):
    task_data = iterate_directory(task_data_main_dir)
    task_data_files = iterate_directory(task_data['directories']['data'])
    if task == "task3a":
        task_data_files = iterate_directory(task_data_files['directories']['task_3A'])
        task_test_data = iterate_directory(task_data['directories']['task_3A_test'])
    elif task == "task4":
        fact_train_file_path = "data/task_4/task_4_news_media_factuality_train.tsv"
        fact_dev_file_path = "data/task_4/task_4_news_media_factuality_dev.tsv"
        fact_test_gold_file_path = "task_4_news_media_factuality_test.tsv"

        fact_train_raw_data = pd.read_csv(os.path.join(task_data_main_dir, fact_train_file_path), sep = "\t")
        fact_dev_raw_data = pd.read_csv(os.path.join(task_data_main_dir, fact_dev_file_path), sep = "\t")
        fact_test_raw_data = pd.read_csv(os.path.join(task_data_main_dir, fact_test_gold_file_path), sep = "\t")

        fact_train_data = get_split_data(fact_train_raw_data, task_data_main_dir)
        fact_dev_data  = get_split_data(fact_dev_raw_data, task_data_main_dir)
        fact_test_data = get_split_data(fact_test_raw_data, task_data_main_dir)

        df_train = pd.DataFrame({
            'content': fact_train_data[0],
            'label': fact_train_data[1]
        })

        df_dev = pd.DataFrame({
            'content': fact_dev_data[0],
            'label': fact_dev_data[1]
        })

        df_test = pd.DataFrame({
            'content': fact_test_data[0],
            'label': fact_test_data[1]
        })


    if task == "task3a":
        # Load data from the directories into Dataset objects
        train_dataset = load_data_from_directory(task_data_files['directories']['train_json'])
        dev_dataset = load_data_from_directory(task_data_files['directories']['dev_json'])
        test_dataset = load_data_from_directory(task_test_data['directories']['test_json'])
    elif task == "task4":
        train_dataset = Dataset.from_pandas(df_train)
        dev_dataset = Dataset.from_pandas(df_dev)
        test_dataset = Dataset.from_pandas(df_test)

    return train_dataset, dev_dataset, test_dataset


def create_label_maps(train_dataset):
    label2id = {data_point['label_text']: data_point['label'] for data_point in train_dataset}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label