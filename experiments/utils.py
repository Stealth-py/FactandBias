import pandas as pd


def format_metrics_df(metrics_df):
    # Extract the 'accuracy', 'macro avg', and label-level metrics from train_report
    train_accuracy = metrics_df['train_report'].apply(lambda x: x['accuracy'])
    dev_accuracy = metrics_df['dev_report'].apply(lambda x: x['accuracy'])
    test_accuracy = metrics_df['test_report'].apply(lambda x: x['accuracy'])

    train_macro_avg_f1 = metrics_df['train_report'].apply(lambda x: x['macro avg']['f1-score'])
    dev_macro_avg_f1 = metrics_df['dev_report'].apply(lambda x: x['macro avg']['f1-score'])
    test_macro_avg_f1 = metrics_df['test_report'].apply(lambda x: x['macro avg']['f1-score'])

    train_weighted_avg_f1 = metrics_df['train_report'].apply(lambda x: x['weighted avg']['f1-score'])
    dev_weighted_avg_f1 = metrics_df['dev_report'].apply(lambda x: x['weighted avg']['f1-score'])
    test_weighted_avg_f1 = metrics_df['test_report'].apply(lambda x: x['weighted avg']['f1-score'])

    # train_label_metrics = pd.DataFrame(metrics_df['train_report'].tolist()).drop(columns=['accuracy', 'macro avg', 'weighted avg'])

    ## add columns for dev and test as well

    # Add the extracted metrics to the metrics_df DataFrame
    metrics_df['train_accuracy'] = train_accuracy
    metrics_df['dev_accuracy'] = dev_accuracy
    metrics_df['test_accuracy'] = test_accuracy

    metrics_df['train_macro_avg_f1'] = train_macro_avg_f1
    metrics_df['dev_macro_avg_f1'] = dev_macro_avg_f1
    metrics_df['test_macro_avg_f1'] = test_macro_avg_f1

    metrics_df['train_weighted_avg_f1'] = train_weighted_avg_f1
    metrics_df['dev_weighted_avg_f1'] = dev_weighted_avg_f1
    metrics_df['test_weighted_avg_f1'] = test_weighted_avg_f1

    # metrics_df[train_label_metrics.columns] = train_label_metrics

    return metrics_df