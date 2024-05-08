# Model Editing and Unlearning
## Overview

The `model_editing_and_unlearning` folder contains only this README as we used external repositories as detailed below.

Model Unlearning refers to a set of techniques to remove the influence of specific training examples from the weights of a trained model without retraining the model from scratch. This could be used, for example, when somebody wishes to practise their Right to be Forgotten, removing private data from the training data after model training.

Model Editing refers to updating a model’s behaviour with regard to a specific edit descriptor by updating, erasing or inserting knowledge. Machine Editing techniques overlap with Machine Unlearning but are more focused on changing or erasing a specific ``fact" (e.g. the president of the US is Joe Biden) as opposed to forgetting certain training data instances.

For more detail on machine unlearning and editing to the [report](../reports).

## Code Repositories

The table below summarizes code repositories we investigated, the types of LM they work on out-the-box and the maturity refers to how easily usable we found the repo in terms of documentation and code. We ran experiments using the Knowledge Neurons Method to our pre-trained models with minimal changes to the repository. For detail on our findings please refer to the [report](../reports).

| Approach        | Maturity | MLM  | Decoder | Encoder-Decoder |
|-------------------------------------------------------------------|----------|------| --------| ---------|
| [Knowledge Neurons](https://github.com/EleutherAI/knowledge-neurons) | High     | [x] | [x]    | [x]      |
| [Knowledge Editing](https://github.com/nicola-decao/KnowledgeEditor) | Low      | [x]  | [x]    | [x]      |
| [ROME](https://github.com/kmeng01/rome}{ROME}~\cite{meng2022locating) | High     |      | [x]    |                  |
| [MEMIT](https://github.com/kmeng01/memit)                         | High     |      | [x]   |                  |
| [MEND](https://github.com/eric-mitchell/mend)                     | Low      |      | [x]    |                  |
| [Knowledge Unlearning](https://github.com/joeljang/knowledge-unlearning) | High     |      | [x]  |                  |
