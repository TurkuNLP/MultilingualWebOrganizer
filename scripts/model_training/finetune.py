from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_from_disk, Dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from pathlib import Path
import json
import argparse
import logging


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    return logger


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


def init_base_model(model_name: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return tokenizer, model


def load_jsonl(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_data(dataset_path: str):
    return load_from_disk(dataset_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a multilingual BERT model on a classification task."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="jhu-clsp/mmBERT-base",
        help="Pretrained model name or path.",
    )
    parser.add_argument(
        "--do-train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=24,
        help="Number of labels for classification.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the Hugging Face dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save the model and checkpoints.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5, help="Learning rate for training."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay for optimizer."
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="epoch",
        help="Evaluation strategy (e.g., 'steps', 'epoch').",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        help="Checkpoint saving strategy (e.g., 'steps', 'epoch').",
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        help="Load the best model at the end of training.",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="f1",
        help="Metric to use for selecting the best model.",
    )
    return parser.parse_args()


def main(args):
    logger = init_logger()
    model_name = args.model_name
    num_labels = args.num_labels

    dataset = load_data(args.dataset_path)

    logger.info("Dataset loaded successfully.")
    logger.info(f"Dataset splits: {dataset.keys()}")
    logger.info(f"Number of training samples: {len(dataset['train'])}")
    logger.info(f"Number of validation samples: {len(dataset['validation'])}")
    logger.info(f"Number of test samples: {len(dataset['test'])}")
    logger.info(f"Example training sample: {dataset['train'][0]}")

    def tokenize_function(examples):
        return tokenizer(examples, truncation=True, padding=True, max_length=8192)

    train_dataset = dataset["train"].map(tokenize_function, batched=True)
    eval_dataset = dataset["validation"].map(tokenize_function, batched=True)
    test_dataset = dataset["test"].map(tokenize_function, batched=True)

    if args.do_train:
        if args.metric_for_best_model in ["accuracy", "f1"]:
            greater_is_better = True
        else:
            greater_is_better = False
        tokenizer, model = init_base_model(model_name, num_labels=24)
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
            evaluation_strategy=args.evaluation_strategy,
            save_strategy=args.save_strategy,
            load_best_model_at_end=args.load_best_model_at_end,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=greater_is_better,
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
