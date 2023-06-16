"""
Trains a text classification model using device-agnostic code to predict three classes: left / center / right
"""

from pathlib import Path
import pprint
from datasets import Dataset, load_dataset
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import argparse
# import wandb

import data_setup, engine


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a text classification model")
    parser.add_argument("--task", type=str, choices=["task3a", "task3b", "task4"], help="Task name")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained model name")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for optimizer")
    args = parser.parse_args()
    return args

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Print the parsed arguments
    print("Training Configuration:")
    pprint.pprint(vars(args))

    # Setup hyperparameters
    TASK = args.task
    MODEL_NAME = args.model_name
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate


    # Setup the target device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("CUDA not found. Cannot connect to GPU.")

    num_labels = 3  # Number of labels for classification
    
    if TASK == "task3a":
        # Setup data directory for CLEF task 03a
        task_data_main_dir = Path("../clef2023-checkthat-lab/task3/data")

    elif TASK == "task4":
        # Setup data directory for CLEF task 4
        task_data_main_dir = Path("../clef2023-checkthat-lab/task4/data")

    # Load data from the directories into Dataset objects
    train_dataset, dev_dataset, test_dataset = data_setup.create_dataloaders(task_data_main_dir, TASK)

    if TASK == "task3a":
        # Creating label to id and id to label maps
        label2id, id2label = data_setup.create_label_maps(train_dataset)

    elif TASK == "task4":
        # Creating label to id and id to label maps
        label2id = {'0': 'low', '1': "mixed",'2': 'high'}
        id2label = {i: label for label, i in label2id.items()}

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, label2id=label2id, id2label=id2label)
    model.to(device)

    # Define loss function
    loss_function = torch.nn.L1Loss()
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    engine.train(task=TASK, 
                 model=model, 
                model_name=MODEL_NAME, 
                tokenizer=tokenizer, 
                train_loader=train_loader, 
                dev_loader=dev_loader, 
                test_loader=test_loader, 
                loss_function=loss_function,
                optimizer=optimizer, 
                learning_rate=LEARNING_RATE,
                num_epochs=NUM_EPOCHS,
                num_labels=num_labels, 
                device=device)
    

if __name__ == "__main__":
    main()