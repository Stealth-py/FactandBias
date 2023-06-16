import torch
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
import utils
import os


# Define training and evaluation functions
def train_epoch(model, data_loader, loss_function, optimizer, tokenizer, device, num_labels):
    model.train()
    total_loss = 0.0
    for batch in tqdm(data_loader):
        inputs = tokenizer.batch_encode_plus(batch['content'], padding=True, truncation=True, return_tensors='pt')
        labels = batch['label'].to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        optimizer.zero_grad()
        outputs = model(**inputs)
        
        # print(outputs)
        # print(torch.argmax(outputs.logits.squeeze(), dim=1).float(), labels.float())

        # softmax to apply on dimension 1, i.e. per row
        ## Check if I should use logits or softmax in loss_fn ##
        # logsoftmax = torch.nn.Softmax(dim=1)
        # predicted_labels = logsoftmax(outputs.logits.squeeze())
        # print(predicted_labels)

        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_labels).to(device)

        # Calculate the loss using L1Loss
        loss = loss_function(outputs.logits.squeeze(), one_hot_labels.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, loss_function, tokenizer, device, num_labels):
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs = tokenizer.batch_encode_plus(batch['content'], padding=True, truncation=True, return_tensors='pt')
            labels = batch['label'].to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_labels).to(device)

            ## Check if I should use logits or softmax in loss_fn ##
            loss = loss_function(outputs.logits.squeeze(), one_hot_labels.float())
            total_loss += loss.item()

            _, predicted_labels = torch.max(outputs.logits, dim=1)
            predictions.extend(predicted_labels.tolist())
            true_labels.extend(labels.tolist())

    return total_loss / len(data_loader), predictions, true_labels



def train(task,
          model, 
          model_name, 
          tokenizer, 
          train_loader, 
          dev_loader, 
          test_loader, 
          loss_function, 
          optimizer,
          learning_rate,
          num_epochs, 
          num_labels, 
          device):

    # Create pandas dataframe to store metrics
    # metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'dev_loss', 'test_loss', 'train_report', 'dev_report', 'test_report'])

    # Create an empty list to store metrics
    metrics_list = []
    best_dev_macro_avg_f1 = -1  # Initialize with a low value

    # replacing '/' with '-' in model_name
    model_name = model_name.replace("/", "-")

    print(f"Starting model training for CLEF 2023 {task}.")

    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, loss_function, optimizer, tokenizer, device, num_labels)
        
        _ , train_predictions, train_true_labels = evaluate_model(model, train_loader, loss_function, tokenizer, device, num_labels)
        
        dev_loss, dev_predictions, dev_true_labels = evaluate_model(model, dev_loader, loss_function, tokenizer, device, num_labels)
        
        test_loss, test_predictions, test_true_labels = evaluate_model(model, test_loader, loss_function, tokenizer, device, num_labels)

        train_report = classification_report(train_true_labels, train_predictions, output_dict=True)
        dev_report = classification_report(dev_true_labels, dev_predictions, output_dict=True)
        test_report = classification_report(test_true_labels, test_predictions, output_dict=True)

        # Store metrics in dataframe
        metrics_list.append({'epoch': epoch,
                            'train_L1_loss': train_loss,
                            'dev_L1_loss': dev_loss,
                            'test_L1_loss': test_loss,
                            'train_report': train_report, 
                            'dev_report': dev_report,
                            'test_report': test_report
                            })

        # Print metrics for current epoch
        print(f"Epoch {epoch}:")
        print(f"Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | Test Loss: {test_loss:.4f}")
        print(f"Macro Avg F1 score | Train:{train_report['macro avg']['f1-score']} | Dev:{dev_report['macro avg']['f1-score']} | Test:{test_report['macro avg']['f1-score']}")

        # Check if current epoch has the best dev_macro_avg_f1
        dev_macro_avg_f1 = dev_report['macro avg']['f1-score']
        if dev_macro_avg_f1 > best_dev_macro_avg_f1:
            # Update best_dev_macro_avg_f1
            best_dev_macro_avg_f1 = dev_macro_avg_f1
            # Save the model checkpoint
            output_dir = f'../best_models/{task}/{model_name}/{learning_rate}/'
            os.makedirs(output_dir, exist_ok=True)
            model_checkpoint_path = os.path.join(output_dir, f'best_model_{model_name}.pt')
            model.save_pretrained(model_checkpoint_path)
            tokenizer.save_pretrained(model_checkpoint_path)
            print(f"Saved best model, at epoch {epoch}.")


    # Convert metrics list to DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    metrics_df = utils.format_metrics_df(metrics_df)

    metrics_df['task'] = task
    metrics_df['model_name'] = model_name
    metrics_df['learning_rate'] = learning_rate

    # Save metrics dataframe to a JSON file
    os.makedirs(f'../results/clef-{task}/', exist_ok=True)
    metrics_df.to_json(f'../results/clef-{task}/training_results-{model_name}-{learning_rate}.json', orient='records', lines=True, indent=2)
    metrics_df.to_csv(f'../results/clef-{task}/training_results-{model_name}-{learning_rate}.csv')