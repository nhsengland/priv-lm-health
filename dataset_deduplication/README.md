# Training Data Deduplication
## Overview

Please note the dataset_deduplication folder contains only this `README.md` as we used external packages for dataset deduplication as detailed below.

Training Data deduplication has been suggested as an effective measure to prevent training data leakage from large LMs, offering privacy guarantees across the dataset. Training datasets for LMs are typically deduplicated at the document level e.g. removing repeated webpages. However, text sequences are often duplicated both within and across different documents. Recent studies have shown that eliminating duplicate sequences from the training data, utilizing a suffix array-based algorithm, results in GPT-2 LMs generating approximately 10x less training data[[1]](#1). Expanding on this, empirical evidence indicates that the likelihood of GPT-Neo LM generating exact sequences from the training data increases in proportion to the presence of duplicates in the training data[[2]](#2).

In this work, we explored 3 deduplication methods: <br>
(1) Near Duplication with Minhashing: aims to remove full training examples or documents with high token n-gram overlap. <br>
(2) Exact Substring Deduplication with a suffix array based algorithm[[1]](#1): looks for text sequences of a specified minimum length that overlap highly in the training data and removes all but one of these. <br>
(2) Semantic Deduplication using pre-trained language model embeddings[[3]](#3): aims to find sentences with high semantic similarity and removes all but one of these. <br>

Please refer to the technical report, located in the `reports` section of this repository, for more detailed explanations on each of these methods and our findings.

## Code Repositories Used
- [text-dedup](https://github.com/ChenghaoMou/text-dedup) provides an easy-to-use package for a number of deduplication techniques including min-hashing and suffix array-based deduplication [[1]](#1), which we adapted with minimal changes to run over the MIMIC-III Notes.
- [BertTopic](https://maartengr.github.io/BERTopic/index.html) provides a comprehensive package for Topic Modelling, which we used for semantic clustering and to visualise clusters, in order to sanity check them as the initial step for semantic deduplication. Unfortunately, we did not find a package for the end-to-end process of semantic deduplication.

## References
<a id="1">[1]</a>
Lee, Katherine, et al. "Deduplicating training data makes language models better." arXiv preprint arXiv:2107.06499 (2021). <br>
<a id="2">[2]</a>
Carlini, Nicholas, et al. "Quantifying memorization across neural language models." arXiv preprint arXiv:2202.07646 (2022). <br>
<a id="3">[3]</a>
Abbas, Amro, et al. "SemDeDup: Data-efficient learning at web-scale through semantic deduplication." arXiv preprint arXiv:2303.09540 (2023). <br>
