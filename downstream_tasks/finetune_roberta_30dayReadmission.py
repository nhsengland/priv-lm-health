"""
This script id for fine-tuning any masked language model on the MIMIC-III 30-day
Readmission Task. The default hyperparameters are what we used for fine-tuning
RoBERTa-base and we would recommend using the same ones if you are tuning this model.
This notebook was adapted from this tutorial:
https://huggingface.co/docs/transformers/training. Thank you!
"""

import os

import evaluate
import pandas as pd
import torch
import typer
from datasets import Dataset
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoTokenizer,
)

from privlm.utils import write_jsonl

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(
    model_ckpt: str = typer.Option(
        default="roberta-base",
        help=(
            "Model Checkpoint to Fine-tune on the Readmission"
            " task- must be compatible with huggingface Transformers"
        ),
    ),
    data_path: str = typer.Option(
        default="../datasets/readmission_task/",
        help=(
            "the location of your saved training"
            " and test data for the readmission task"
        ),
    ),
    num_train_epochs: int = typer.Option(
        default=5, help="number of epochs to train the model for"
    ),
    learning_rate: int = typer.Option(
        default=1e-5, help="learning rate to use for training"
    ),
    train_batch_size: int = typer.Option(
        default=16, help="batch size to use for training"
    ),
    gradient_accumulation_steps: int = typer.Option(
        default=2, help="gradient accumulation to use for training"
    ),
    weight_decay: int = typer.Option(
        default=0.01, help="weight decay to use for training"
    ),
):
    ########## SET VARS ############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    log_dir_name = f"./logs/readmission/{model_ckpt}/"
    results_dir_name = f"./results/models/readmission/{model_ckpt}/"
    results_save_file_path = "./results/scores/readmission/model_ckpt.json"

    ########## READ IN DATA ############
    raw_train_df = pd.read_csv(str(data_path) + "train.csv")[["TEXT", "Label"]]
    train_df = raw_train_df.rename(columns={"TEXT": "text", "Label": "label"})
    train_df = train_df.astype({"text": str, "label": int})
    print(f"""Train Label Counts: {train_df["label"].value_counts()}""")
    train = Dataset.from_pandas(train_df)

    raw_valid_df = pd.read_csv(str(data_path) + "val.csv")[["TEXT", "Label"]]
    valid_df = raw_valid_df.rename(columns={"TEXT": "text", "Label": "label"})
    valid_df = valid_df.astype({"text": str, "label": int})
    print(f"""Val Label Counts: {valid_df["label"].value_counts()}""")
    valid = Dataset.from_pandas(valid_df)

    raw_test_df = pd.read_csv(str(data_path) + "test.csv")[["TEXT", "Label"]]
    test_df = raw_test_df.rename(columns={"TEXT": "text", "Label": "label"})
    test_df = test_df.astype({"text": str, "label": int})
    print(f"""Test Label Counts: {test_df["label"].value_counts()}""")
    test = Dataset.from_pandas(test_df)

    ############### TOKENIZE DATA #################
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_train = train.map(tokenize_function, batched=True)
    tokenized_val = valid.map(tokenize_function, batched=True)
    tokenized_test = test.map(tokenize_function, batched=True)

    ############## SETUP MODEL FOR TEXT CLASSIFICATION ##################
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            model_ckpt, num_labels=2
        )

    ###############  SET EVAL METRIC & TRAINING ARGS ###############

    def compute_auroc(eval_preds):
        logits, labels = eval_preds
        preds = softmax(logits, axis=1)[:, 1]
        auroc = roc_auc_score(y_true=labels, y_score=preds)
        return {"auroc": auroc}

    # train
    training_args = TrainingArguments(
        output_dir=results_dir_name,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        warmup_steps=100,
        logging_dir=log_dir_name,
        load_best_model_at_end=True,
        metric_for_best_model="auroc",
        report_to="tensorboard",
        save_total_limit=2,
        fp16=True,
    )

    ###############  TRAIN THE MODEL ###############

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_auroc,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=1, early_stopping_threshold=0.0
            )
        ],
    )

    trainer.train()

    ###############  EVALUATE THE MODEL ON TEST SET ###############

    outputs = trainer.predict(tokenized_test)
    predictions = softmax(outputs.predictions, axis=1)[:, 1]
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    clf_results = clf_metrics.compute(
        predictions=predictions, references=outputs.label_ids
    )
    auroc = roc_auc_score(y_true=outputs.label_ids, y_score=predictions)
    clf_results["aucroc"] = auroc
    print(clf_results)
    # save results to file
    write_jsonl(file_path=results_save_file_path, lines=[clf_results])


if __name__ == "__main__":
    typer.run(main)
