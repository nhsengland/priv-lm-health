# Investigating Privacy Risks and Mitigations in Healthcare Language Models
## NHS England DART - PhD Internship Project

### About the Project

[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This `priv-lm-health` repository holds code for Investigating Privacy Risks and Mitigations in Healthcare Large Language Models project. This work presents an initial exploration of the privacy risk landscape for training data used to train healthcare Language Models (LMs) and various privacy-preserving techniques, applied to LMs both before and after model training. We evaluate their effectiveness with privacy attacks.

:warning: **This repository is experimental and thus models generated and attacked using this repository are not suitable to deploy into a production environment without further testing and evaluation.** :warning:

This work was conducted as part of an NHS England DART PhD Internship project by [Victoria Smith](https://github.com/v-smith) between June and November 2023.

Further information on the original project proposal can be found [here](https://nhsx.github.io/nhsx-internship-projects/language-model-privacy-leakage/) and a case study is presented [here](https://nhsengland.github.io/datascience/articles/2024/04/11/privLM/).

The associated report can be found in the [reports](./reports) folder.

_**Note:** No data are shared in this repository._

### Project Structure

- The main code is found in the root of the repository (see Usage below for more information).
- The accompanying report is also available in the [reports](./reports) folder.
- More information about the code usage can be found in the [model card](./model_card.md)

```
├── dataset_deduplication # Information on dataset deduplication techniques and packages.
├── downstream_tasks # Pipelines for fine-tuning/evaluating LMs on downstream text classification tasks.
├── membership_inference # Code and notebooks for comparing and visualising LM embeddings.
├── model_editing_and_unlearning  # Information on model editing and unlearning packages.
├── privlm # package containing reusable functions for privacy attacks in LMs, and loading data.
├── model_pretraining # Information on preparing data and pre-training language models.
├── reports # Project reports
├── .gitignore
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENCE
├── OPEN_CODE_CHECKLIST.md
├── README.md
├── requirements.txt
└── setup.py
```

#### `downstream_tasks`
A set of pipelines for fine-tuning and evaluating text classification models on downstream tasks- GLUE and MIMIC-III 30-day readmission. More details in the [report](./reports).

#### `membership_inference`
A set of scripts for running Loss and Likelihood Ratio Membership Inference Attacks on Masked Language Models, and quantifying the privacy risk of the models.

### Built With

[![Python v3.9](https://img.shields.io/badge/python-v3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

The work is mostly undertaken utilising the [`Transformers`](https://huggingface.co/docs/transformers/index) library for language modelling.

### Getting Started

#### Installation

To get a local copy up and running follow these simple steps.

To clone the repo:

`git clone https://github.com/nhsengland/priv-lm-health`

To create a suitable environment:
- ```conda create --name priv-lm-health python==3.9```
- `conda activate priv-lm-health`
- `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
- `pip install -e .`

#### GPU Support

Training with GPU is recommended. Single-GPU training has been tested with:
- `NVIDIA Tesla T4`
- `cuda 11.3`
- `Linux Server`

The repository uses [pre-commit](https://pre-commit.com) hooks to enforce code style using [Black](https://github.com/psf/black), follows [flake8](https://github.com/PyCQA/flake8), and performs a few other checks.  See `.pre-commit-config.yaml` for more details. These hooks will also need installing locally via:

```{bash}
pre-commit autoupdate
pre-commit install
```

and then will be checked on commit.

### Usage

__NOTE__ The language modelling and membership inference code in this repo can be used out of the box. Please take care with how the directories are setup and attention to how these different files are saved.

To fine-tune models on MIMIC-III 30-day readmission task or GLUE tasks, refer to the `downstream_tasks` folder. Please note the MIMIC-III 30-day readmission task requires dataset preprocessing. Refer to the folder [README.md](downstream_tasks%2FREADME.md) for details.

To perform membership inference attacks on pre-trained or fine-tuned LMs, please refer to the `membership_inference` folder. You will initially need to prepare the data with the [prepare_data_MI.py](./membership_inference/prepare_data_MI.py) script before running the attack. You will likely need to use gpu when running the attack code.

#### Outputs
The associated report can be found in the [reports](./reports) folder.

#### Datasets

##### Psuedo-Reidentified MIMIC-III data

You need to apply for a licence to use this dataset and once you have it, it can be downloaded from [Physionet](https://www.physionet.org/content/clinical-bert-mimic-notes/1.0.0/). We used this data for pre-training our models, fine-tuning our models on the 30-day readmission task and for our member and non-member datasets in our Membership Inference Attacks.

##### i2b2 2014_risk_factors

We used data from the `n2c2_2014_risk_factors` dataset which was part of the i2b2 challenge and can be downloaded from the [DBMI Data Portal](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/). Please note you also need to apply for access to download this data. We only used this data as an external dataset in our Membership Inference Attacks. It was not used for any Language Modelling.

### Roadmap

See the repo [Issues](./Issues/) for a list of proposed features (and known problems).

### Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

### License

Unless stated otherwise, the codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

_See [LICENSE](./LICENSE) for more information._

The documentation is [© Crown copyright][copyright] and available under the terms
of the [Open Government 3.0][ogl] licence.

[mit]: LICENCE
[copyright]: http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/
[ogl]: http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

### Contact

To find out more get in touch at [datascience@nhs.net](mailto:datascience@nhs.net).
