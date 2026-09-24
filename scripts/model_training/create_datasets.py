import datasets
import json
import numpy as np  # type: ignore
from pathlib import Path
import datetime

np.random.seed(42)


def read_jsonl(file_path: Path):
    data = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def shuffle_data(data: list[dict]):
    np.random.shuffle(data)
    return data


def create_splits(
    data: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    train_size = int(train_ratio * len(data))
    val_size = int(val_ratio * len(data))

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]

    return train_data, val_data, test_data


def create_dataset_dict(
    train_data: list[dict], val_data: list[dict], test_data: list[dict]
):
    dataset_dict = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_list(train_data),
            "validation": datasets.Dataset.from_list(val_data),
            "test": datasets.Dataset.from_list(test_data),
        }
    )
    return dataset_dict


def save_dataset_dict(
    dataset_dict: datasets.DatasetDict, output_dir: Path, readme_content: str = None
):
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    # Save a README file with information about the dataset
    readme_path = output_dir / "README.md"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme_content)


def make_readme(args):
    if Path(args.input).is_dir():
        input_files = list(Path(args.input).glob("*.jsonl"))
        input_files_str = "- " + "\n- ".join(str(f) for f in input_files)
    else:
        input_files_str = str(Path(args.input))

    readme_content = f"""
# Dataset README

This dataset was created on {datetime.datetime.now().strftime("%Y-%m-%d")}.

## Input Files
The dataset was created from the following input files:
{input_files_str}
    
## Splits
The dataset is split into train, validation, and test sets with the following ratios:
- Train: {args.train_ratio}
- Validation: {args.val_ratio}
- Test: {args.test_ratio}
    
## Creation Script
The dataset was created using the script: create_datasets.py
    """
    return readme_content.strip()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create train, validation, and test datasets from a JSONL file."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input JSONL file or directory of JSONL files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Ratio of training data."
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Ratio of validation data."
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1, help="Ratio of test data."
    )
    return parser.parse_args()


def validate_args(args):
    if args.train_ratio + args.val_ratio + args.test_ratio != 1.0:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    if args.train_ratio < 0 or args.val_ratio < 0 or args.test_ratio < 0:
        raise ValueError("Train, validation, and test ratios must be non-negative")
    if not args.input:
        raise ValueError("Input path must be provided")
    if not args.output_dir:
        raise ValueError("Output directory path must be provided")
    if args.input and not Path(args.input).exists():
        raise FileNotFoundError(f"Input path {args.input} does not exist")


def main():
    args = parse_args()
    validate_args(args)

    input_path = Path(args.input)
    if input_path.is_dir():
        jsonl_files = list(input_path.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found in directory {input_path}")
        data = []
        for file in jsonl_files:
            data.extend(read_jsonl(file))
    else:
        data = read_jsonl(input_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory {output_dir} already exists and is not empty. Use --overwrite to overwrite."
            )

    # Shuffle the data and create splits
    shuffled_data = shuffle_data(data)
    train_data, val_data, test_data = create_splits(
        shuffled_data, args.train_ratio, args.val_ratio, args.test_ratio
    )
    dataset_dict = create_dataset_dict(train_data, val_data, test_data)

    # Save dataset with a README file containing information about the dataset
    readme_content = make_readme(args)
    save_dataset_dict(dataset_dict, output_dir, readme_content)


if __name__ == "__main__":
    main()
