from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification, 
    TrainingArguments,
    Trainer
)
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from pathlib import Path
import json
import argparse

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted')
    }
    
def init_base_model(model_name: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return tokenizer, model

def load_jsonl(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_data():
    train_data = load_jsonl(Path("data/train.jsonl"))
    eval_data = load_jsonl(Path("data/eval.jsonl"))
    test_data = load_jsonl(Path("data/test.jsonl"))
    
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(eval_data),
        "test": Dataset.from_list(test_data)
    })
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a multilingual BERT model on a classification task.")
    parser.add_argument("--model_name", type=str, default="jhu-clsp/mmBERT-base", help="Pretrained model name or path.")
    parser.add_argument("--do-train", action="store_true", help="Whether to run training.")
    parser.add_argument("--num_labels", type=int, default=24, help="Number of labels for classification.")
    parser.add_argument("--train_file", type=str, default="data/train.jsonl", help="Path to the training data file.")
    parser.add_argument("--eval_file", type=str, default="data/eval.jsonl", help="Path to the evaluation data file.")
    parser.add_argument("--test_file", type=str, default="data/test.jsonl", help="Path to the test data file.")
    parser.add_argument("--output_dir", type=str, default="./mmbert-xnli", help="Directory to save the model and checkpoints.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer.")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="Evaluation strategy (e.g., 'steps', 'epoch').")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Checkpoint saving strategy (e.g., 'steps', 'epoch').")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load the best model at the end of training.")
    parser.add_argument("--metric_for_best_model", type=str, default="f1", help="Metric to use for selecting the best model.")
    return parser.parse_args()

def main(args):
    model_name = args.model_name
    num_labels = args.num_labels
    
    dataset = load_data()
    
    def tokenize_function(examples):
        texts = [f"{p} {tokenizer.sep_token} {h}" 
                for p, h in zip(examples["premise"], examples["hypothesis"])]
        
        return tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=8192
        )
    
    train_dataset = dataset["train"].map(tokenize_function, batched=True)
    eval_dataset = dataset["validation"].map(tokenize_function, batched=True)
    test_dataset = dataset["test"].map(tokenize_function, batched=True)
    
    if args.do_train:
        tokenizer, model = init_base_model(model_name, num_labels=24)
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            learning_rate=3e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        
    else:
        # Load the model from the output directory
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        trainer = Trainer(
            model=model,
            compute_metrics=compute_metrics,
        )
        
        # Evaluate on the test set
        test_results = trainer.evaluate(test_dataset)
        print("Test results:", test_results)

if __name__ == "__main__":
    args = parse_args()
    main(args)
