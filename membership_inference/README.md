# Membership Inference Attacks
## Overview

The `membership_inference` folder contains scripts for running membership inference attacks (MIAs) on any masked language models (MLMs) and metrics to quantify privacy risk of the attacks.

MIAs are a type of privacy attacks which tries to deduce whether a specific data point was part of a target-model's training dataset, as described in [[1]](#1). Please see the [report](../reports) for more detailed information on MIA types and risks.

We applied these MIAs to our pre-trained RoBERTa models, which were trained on MIMIC-III Notes with different data deduplication strategies (see [dataset_deduplication](../dataset_deduplication)), to quantify the memorization of training data in the models.

## Preparing the Attack Datasets

The [prepare_data_MI.py](./prepare_data_MIA.py) file can be used for generating and saving the following three datasets needed for the MIAs. For all datasets we used a random selection of 1000 sentences from 90 patients.

### Members
We selected data from the 50% split of the [psuedo-reidentified MIMIC-III notes](https://www.physionet.org/content/clinical-bert-mimic-notes/1.0.0/), which we used for pre-training our models.

### Non-members
We selected data from the 50% split of the [psuedo-reidentified MIMIC-III notes](https://www.physionet.org/content/clinical-bert-mimic-notes/1.0.0/), which we did not use for pre-training our models. We also ensured that there was no overlap in patients with the 50% data used for training, due the the high overlap in text between notes from the same patients.

### Externals
We used data from the `n2c2_2014_risk_factors` dataset which was part of the i2b2 challenge and can be downloaded from the [DBMI Data Portal](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/). Please note you need to apply for access to download this data.

## Implementing the Attack

The [score_membership_inference.py](./score_MIA.py) file can be used to run the attacks and get output privacy risk scores. We implemented the attack code following the methodology for Likelihood Ratio MIAs and the Baseline Loss-based MIAs from [[2]](#2). There was no repository already available with this paper. We used the [mincons](https://github.com/kanishkamisra/minicons) package to calculate the log likelihood of a single sequence passed to our model. Please note to run this attack you will need to use gpu due to memory constraints.

## References
<a id="1">[1]</a>
Shokri, Reza, et al. "Membership inference attacks against machine learning models." 2017 IEEE symposium on security and privacy (SP). IEEE, 2017.
<br>
<a id="2">[2]</a>
Mireshghallah, Fatemehsadat, et al. "Quantifying privacy risks of masked language models using membership inference attacks." arXiv preprint arXiv:2203.03929 (2022).
