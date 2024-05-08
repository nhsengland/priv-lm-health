# Downstream Tasks
## Overview

The `downstream_tasks` folder contains code for the training and evaluation of pre-trained RoBERTa models on a selection on the MIMIC-III 30 day readmission task[[1]](#1) and the [GLUE Benchmark](https://gluebenchmark.com/) datasets as a general NLP benchmark.

These downstream tasks were intended as a sanity check on our pre-trained models (see the `model_pretraining` folder), and did not serve a further purpose as this work was focused on privacy.

## Preparing Datasets
### MIMIC-III 30 day Readmission Data

To prepare the MIMIC-III notes for the 30-day readmission task, we used the `preprocess.py` from the original [Clinical Bert GitHub Repo](https://github.com/kexinhuang12345/clinicalBERT) from [[1]](#1), out-the-box.

### GLUE Data

We downloaded the data offline from the [GLUE website](https://gluebenchmark.com/tasks), as we could not access these sites from our virtual machine. To point the correct dataset directories offline, we had to edit the `glue.py` dataset file in both the `data_url` and `data_dir` fields. To use this aspect, you will need to get a copy of the `./glue.py`.

## Model Fine-tuning

 ### MIMIC-III 30 day Readmission Task
The [finetune_roberta.py](./finetune_roberta_30dayReadmission.py) file was constructed using the Trainer from `HuggingFace`. We followed a [tutorial](https://huggingface.co/docs/transformers/training) courtesy of `HuggingFace` to construct the code. Changes focused on saving checkpoints, logs and results locally and writing a new compute_metrics function to allow calculation of ROC AUC.

 ### Baseline GLUE tasks
The [text_classification_glue.py](./finetune_roberta_glue.py) file was constructed from a `HuggingFace` notebook [huggingface-classification-notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb). Changes mainly focused on saving checkpoints, logs and results locally, as well as adaptations to allow the datasets to be loaded offline on our virtual machine.

## References
<a id="1">[1]</a>
 Huang, Kexin, Jaan Altosaar, and Rajesh Ranganath. "Clinicalbert: Modeling clinical notes and predicting hospital readmission." arXiv preprint arXiv:1904.05342 (2019).<br>
