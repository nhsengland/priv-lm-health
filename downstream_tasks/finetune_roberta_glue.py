""" This script id for fine-tuning any masked language model on one of the core GLUE tasks. The default hyperparameters are what we used for fine-tuning RoBERTa-base and we would recommend using the same ones if you are tuning this model. This notebook was adapted from https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb. Thank you! """

import os
from pathlib import Path

import evaluate
import numpy as np
import transformers
import typer
from datasets import load_dataset
from transformers import AutoTokenizer

from privlm.utils import write_jsonl

print(transformers.__version__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(
    task: str = typer.Option(
        default="sst2",
        help=(
            "The GLUE task you want to run from "
            "cola, mnli, qnli, qqp, rte, sst2, stsb, wnli"
        ),
    ),
    local_data: bool = typer.Option(
        default=False, help="Whether to use locally downloaded GLUE data"
    ),
    model_checkpoint: str = typer.Option(
        default="roberta-base",
        help=(
            "Model Checkpoint to Fine-tune on the GLUE task"
            " - must be compatible with huggingface Transformers"
        ),
    ),
    cache_dir: Path = typer.Option(default="./cache"),
    num_train_epochs: int = typer.Option(
        default=5, help="number of epochs to train the model for"
    ),
    learning_rate: int = typer.Option(
        default=2e-5, help="learning rate to use for training"
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
    actual_task = "mnli" if task == "mnli-mm" else task

    if local_data:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        dataset = load_dataset(
            "./glue_local.py", actual_task
        )  # NOTE - you will need to adapt the data_dir and url in glue_local.py
        # to reflect the local path to your data
    else:
        dataset = load_dataset("./glue.py", actual_task)

    metric = evaluate.load("glue", actual_task)

    ############### TOKENIZE DATA #################
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, cache_dir=cache_dir, use_fast=True
    )

    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(
            examples[sentence1_key], examples[sentence2_key], truncation=True
        )

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    ############## SETUP MODEL FOR TEXT CLASSIFICATION ##################
    from transformers import (
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
    # NOTE set output_hidden_states = False - it seems if this is True it will
    # throw an error for the evaluate call of the Trainer.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        output_hidden_states=False,
        cache_dir=cache_dir,
    )

    ############### SETUP CKPT AND MODEL DIRS ##############

    if "saved_models" in model_checkpoint:
        if "declutr" in model_checkpoint:
            model_name = (
                model_checkpoint.split("/")[4] + "_" + model_checkpoint.split("/")[5]
            )
        else:
            model_name = model_checkpoint.split("/")[3]
    else:
        model_name = model_checkpoint.split("/")[-1]

    logging_dir = f"./logs/glue/{task}/{model_name}/5epoch"
    ckpt_dir = f"./results/models/glue/{task}/{model_name}/5epoch"
    results_save_file_path = f"./results/scores/glue_{task}_{model_name}_5epoch.json"

    ###############  SET EVAL METRIC & TRAINING ARGS ###############

    metric_name = (
        "pearson"
        if task == "stsb"
        else "matthews_correlation" if task == "cola" else "accuracy"
    )

    args = TrainingArguments(
        output_dir=f"{ckpt_dir}/",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        push_to_hub=False,
        logging_dir=f"{logging_dir}/",
        save_total_limit=2,
        report_to="all",
    )

    validation_key = (
        "validation_mismatched"
        if task == "mnli-mm"
        else "validation_matched" if task == "mnli" else "validation"
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    ###############  TRAIN THE MODEL ###############

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print(model.device)

    trainer.train()

    ###############  EVALUATE THE MODEL ###############

    print("Evaluating and Saving")
    eval_results = trainer.evaluate()
    print(eval_results)

    ###############  SAVE RESULTS ###############
    write_jsonl(file_path=results_save_file_path, lines=eval_results)


if __name__ == "__main__":
    typer.run(main)
