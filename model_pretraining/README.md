# Model Pre-training
## Overview

The `model_pretraining` folder contains only this README as we used external repository code as detailed below.

We trained and evaluated pre-trained RoBERTa models on the psuedo-reidentified MIMIC-III Notes Dataset [[1]](#1).

## Preparing the Pre-training Datasets

### Psuedo Reidentified MIMIC-III Notes
The data was downloaded from [Physionet](https://www.physionet.org/content/clinical-bert-mimic-notes/1.0.0/), please note you need to apply to access this dataset. #

### Data Deduplication

We created 3 pre-training datasets with different deduplication preprocessing techniques (see [dataset_deduplication](..%2Fdataset_deduplication)) -None, Near Deduplication and Exact Substring Deduplication.

### Data Preprocessing

The data was pre-processed to prepare it to pass into the Language Model using the `prepare_notes_for_lm.py` script. Adaptations were minimal and related to changes to cope with different input data formats and so have not been included in this repository.

## Model Pre-training

We forked and adapted an existing NHS repository: [ELM4PSIR](https://github.com/nhsx/ELM4PSIR) for pre-training our models. We used the `run_lm_pretraining.py` script with the exact same pre-training hyperparameters. Adaptations were minimal and related to saving checkpoints, logs and results and so have not been included in this repository.

## References
<a id="1">[1]</a>
 Lehman, Eric, et al. "Does BERT pretrained on clinical notes reveal sensitive data?." arXiv preprint arXiv:2104.07762 (2021).<br>
