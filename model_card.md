# Model Card: Investigating Privacy Risks and Mitigations in Clinical Language Models

This repository was created as part of an NHS England PhD internship project and contains code to fine-tune and perform Membership Inference Attacks on Clinical Language Models (LM), with the goal of better understanding the risks of training data leakage and exploring possible mitigation strategies.


## Model Details
The implementation of Investigating Privacy Risks and Mitigations in Clinical Language Models within this repository was created as part of an NHS England PhD internship project undertaken by Victoria Smith {LINK TO LAST COMMIT WITH ABBREVIATED SHA}.
## Model Use
### Intended Use
This repository is intended for use in experimenting with privacy preserving approaches for LMs and testing the effect of these on LM privacy risk using Membership Inference Attacks.


### Out-of-Scope Use Cases

This repository is not suitable to evaluate LMs to provide privacy guarantees for production environment.

## Training Data

Experiments in this repository are run against: <br>
- The `Pseudo-reidentified MIMIC-III Notes` from [Physionet](https://www.physionet.org/content/clinical-bert-mimic-notes/1.0.0/).
- The `n2c2_2014_risk_factors` dataset from the [DBMI Data Portal](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/)
Please note both datasets need a licence to download and use.

## Performance and Limitations

The pre-training and fine-tuning are sensitive to hyperparameter choices, depending on the underlying pre-trained model used and datasets selected. Hyperparameter tuning was not possible within the timelines of this project, nor was it the primary focus of the work.

The membership inference attacks are intended to quantify the privacy risk to training data for Masked Language Models under different mitigation scenarios. They are not intended to be used with malicious intent. Testing attacks for wider LM types were not possible within the scope of this project.
