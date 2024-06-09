import torch
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from zeugma.embeddings import EmbeddingTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import AdamW, get_scheduler
from torch.optim.lr_scheduler import LambdaLR
import warnings
warnings.filterwarnings('ignore')

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def clean_and_validate_sentences(data_frame, column_name='sentence'):
    # Remove rows where the sentence is NaN or empty
    data_frame = data_frame.dropna(subset=[column_name])
    data_frame = data_frame[data_frame[column_name].str.strip() != '']
    return data_frame

def default_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)

def train_and_evaluate_clean_model(data_filepath, victim_model, training_args, learning_rate, optimizer_type='AdamW'):
    # Load the data
    orig_ag_data = pd.read_csv(data_filepath, on_bad_lines='skip', sep='\t')

    # Train, validation, test split with shuffling and random seed 42
    orig_ag_data_train, orig_ag_data_val = train_test_split(orig_ag_data, test_size=0.2, random_state=42)
    orig_ag_data_train, orig_ag_data_val = orig_ag_data_train.reset_index(drop=True), orig_ag_data_val.reset_index(drop=True)

    # Extracting sentences and labels
    orig_X_train, orig_y_train = orig_ag_data_train.sentence, orig_ag_data_train.label
    orig_X_val, orig_y_val = orig_ag_data_val.sentence, orig_ag_data_val.label

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(victim_model)
    orig_encoded_X_train = tokenizer(orig_X_train.to_list(), padding='max_length', truncation=True, max_length=64)
    orig_encoded_X_val = tokenizer(orig_X_val.to_list(), padding='max_length', truncation=True, max_length=64)

    # Label encoding
    label_encoder = LabelEncoder()
    orig_encoded_y_train = label_encoder.fit_transform(orig_y_train)
    orig_encoded_y_val = label_encoder.transform(orig_y_val)

    # Dataset creation
    orig_train_dataset = TextDataset(orig_encoded_X_train, orig_encoded_y_train)
    orig_val_dataset = TextDataset(orig_encoded_X_val, orig_encoded_y_val)

    # Model initialization and training
    clf = AutoModelForSequenceClassification.from_pretrained(victim_model, num_labels=4).to('cuda')
    if optimizer_type == 'AdamW':
        optimizer = AdamW(clf.parameters(), lr=learning_rate)

    # Define scheduler
    num_training_steps = len(orig_train_dataset) * training_args.num_train_epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    trainer = Trainer(
        model=clf,
        args=training_args,
        train_dataset=orig_train_dataset,
        eval_dataset=orig_val_dataset,
        optimizers=(optimizer, scheduler)
    )
    trainer.train()

    # Evaluation
    pred = trainer.predict(orig_val_dataset)
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    clean_accuracy = accuracy_score(labels, preds)  # Calculate Clean Accuracy (CA)
    macro_f1 = f1_score(labels, preds, average='macro')
    micro_f1 = f1_score(labels, preds, average='micro')

    # Output metrics
    return {'Clean Accuracy': clean_accuracy, 'Macro F1': macro_f1, 'Micro F1': micro_f1}

def evaluate_model_on_poisoned_data(clean_data_path, poisoned_data_path, victim_model, training_args, learning_rate, optimizer_type='AdamW', backdoor_target_class=0):
    clean_data = pd.read_csv(clean_data_path, on_bad_lines='skip', sep='\t')
    # Load poisoned data
    poisoned_data = pd.read_csv(poisoned_data_path, on_bad_lines='skip', sep='\t')
    poisoned_data['label'] = backdoor_target_class  # Assume backdoor target class

    # Sample and combine with clean data
    poisoned_sample = poisoned_data.sample(n=sample_size, replace=False)
    combined_data = pd.concat([clean_data, poisoned_sample], axis=0).reset_index(drop=True)
    combined_data = clean_and_validate_sentences(combined_data)

    # Split into train and validation sets
    train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42)
    val_data, _ = train_test_split(test_data, test_size=0.5, random_state=42)

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(victim_model)
    encoded_X_train = tokenizer(train_data['sentence'].tolist(), padding='max_length', truncation=True, max_length=64)
    encoded_X_val = tokenizer(val_data['sentence'].tolist(), padding='max_length', truncation=True, max_length=64)

    # Label encoding
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data['label'])
    y_val = label_encoder.transform(val_data['label'])

    # Dataset creation
    train_dataset = TextDataset(encoded_X_train, y_train)
    val_dataset = TextDataset(encoded_X_val, y_val)

    # Model setup and training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(victim_model, num_labels=4).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate) if optimizer_type == 'AdamW' else None

    num_training_steps = len(train_dataset) * training_args.num_train_epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, scheduler)
    )
    trainer.train()

    # Evaluate the model
    results = trainer.predict(val_dataset)
    labels = results.label_ids
    preds = np.argmax(results.predictions, axis=1)

    # Calculate metrics
    metrics = {
        "dataset": poisoned_data_path.split('/')[-1],
        "val acc poisoned": accuracy_score(labels, preds),
        "val macro f1 poisoned": f1_score(labels, preds, average='macro'),
        "val micro f1 poisoned": f1_score(labels, preds, average='micro'),
        "overall trigger rate": np.mean(preds == backdoor_target_class),
        "samples poisoned": dict(zip(val_data.sentence.iloc[:20].tolist(), preds[:20]))
    }

    return metrics


victim_models = [
    'bert-base-uncased',
    'bert-large-uncased',
    'roberta-base',
    'roberta-large',
    'microsoft/deberta-v3-base',
    'distilbert-base-uncased',
    'albert-base-v2',
    'google/electra-small-discriminator'
]

results_folder = 'results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    output_dir='./model_output/',
    save_strategy="no"
)

clean_data_path = "./ag_data/ag_clean.tsv"
ag_data_directory = "./ag_data"
entries = os.listdir(ag_data_directory)
file_list = [entry for entry in entries if os.path.isfile(os.path.join(ag_data_directory, entry)) and entry != "ag_clean.tsv"]
file_list = sorted(file_list)

learning_rate = 5e-5
backdoor_target_class = 0
sample_size = 2000

# Dictionary to hold all metrics for all models
all_model_metrics = {}

# Loop over each victim model
for victim_model in victim_models:
    safe_model_name = victim_model.replace('/', '_')
    model_metrics = {safe_model_name: {}}

    print(f"\nProcessing model: {victim_model}")
    print("Training and evaluating on clean dataset...")

    clean_metrics = train_and_evaluate_clean_model(clean_data_path, victim_model, training_args, learning_rate)
    model_metrics[safe_model_name]['clean_dataset'] = clean_metrics

    print("Finished evaluating clean dataset.")
    print("Starting evaluations on poisoned datasets...")

    # Evaluate on poisoned datasets
    for file in file_list:
        poisoned_data_path = os.path.join(ag_data_directory, file)
        print(f"Evaluating on poisoned data: {file}")
        poisoned_metrics = evaluate_model_on_poisoned_data(clean_data_path, poisoned_data_path, victim_model, training_args, learning_rate, 'AdamW', backdoor_target_class)
        model_metrics[safe_model_name][file] = poisoned_metrics

    print("Finished evaluating all poisoned datasets.")
    # Store metrics for the current victim model in the all-encompassing dictionary
    all_model_metrics[safe_model_name] = model_metrics[safe_model_name]

    # Save individual metrics to a JSON file
    individual_file_path = os.path.join(results_folder, f'{safe_model_name}_metrics.json')
    with open(individual_file_path, 'w') as file:
        json.dump(model_metrics, file, indent=4, default=default_converter)
    print(f"Metrics for {safe_model_name} have been saved to {individual_file_path}")

# Save the comprehensive metrics to a JSON file
metrics_file_path = os.path.join(results_folder, 'all_model_metrics.json')
with open(metrics_file_path, 'w') as file:
    json.dump(all_model_metrics, file, indent=4, default=default_converter)

print(f"All metrics have been saved to {metrics_file_path}")