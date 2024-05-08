"""All functions behind preparing data for and calculating scores for MIAs."""

import copy
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, RobertaForMaskedLM

from privlm import text_scorer as scorer
from privlm.utils import create_batch, file_list_folders, read_jsonl, write_jsonl

random.seed(1)


def get_subject_ids(file_path_list):
    print("GETTING SUBJECT IDS")
    all_subject_ids = []
    for file_path in tqdm(file_path_list):
        data_df = pd.read_csv(file_path)
        subject_ids = list(set(data_df["SUBJECT_ID"].to_list()))
        all_subject_ids.extend(subject_ids)
    all_subject_ids = list(set(all_subject_ids))
    return all_subject_ids


def create_member_df(training_files):
    print("CREATING MEMBER DATAFRAME")
    member_df = pd.DataFrame(columns=["SUBJECT_ID", "TEXT"])
    for file_path in tqdm(training_files):
        tmp_df = pd.read_csv(file_path)[["SUBJECT_ID", "TEXT"]]
        member_df = pd.concat([member_df, tmp_df], ignore_index=True)
        print(len(member_df))
    return member_df


def create_non_member_df(non_training_files, non_member_subject_ids):
    print("CREATING NON-MEMBER DATAFRAME")
    non_member_df = pd.DataFrame(columns=["SUBJECT_ID", "TEXT"])
    for file_path in tqdm(non_training_files):
        df = pd.read_csv(file_path)[["SUBJECT_ID", "TEXT"]]
        tmp_df = df.loc[df["SUBJECT_ID"].isin(non_member_subject_ids)]
        non_member_df = pd.concat([non_member_df, tmp_df], ignore_index=True)
        print(len(non_member_df))
    return non_member_df


def prep_mimic_for_MI(
    input_df, number_of_patients=90, number_of_sentences=4072, min_len_sentences=20
):
    raw_data = input_df[["SUBJECT_ID", "TEXT"]]
    # select patients
    all_subject_ids = list(raw_data["SUBJECT_ID"])
    unique_subject_ids = list(set(all_subject_ids))
    random.shuffle(unique_subject_ids)
    selected_subject_ids = unique_subject_ids[:number_of_patients]
    assert len(selected_subject_ids) == number_of_patients
    # get rows for selected patients
    selected_subject_rows = raw_data.loc[
        raw_data["SUBJECT_ID"].isin(selected_subject_ids)
    ]
    selected_subject_text = list(selected_subject_rows["TEXT"])
    # select sentence
    all_sentences = []
    print("Preparing Sentences")
    for text in tqdm(selected_subject_text):
        list_of_sentences = nltk.sent_tokenize(text)
        all_sentences.extend(list_of_sentences)
    unique_sentences = list(set(all_sentences))
    random.shuffle(unique_sentences)
    long_sentences = [x for x in unique_sentences if len(x) >= min_len_sentences]
    selected_sentence_for_MI = long_sentences[:number_of_sentences]
    assert len(selected_sentence_for_MI) == number_of_sentences
    return selected_sentence_for_MI


def prep_i2b2_for_MI(
    input_dir_path,
    number_of_patients=90,
    number_of_sentences=4072,
    min_len_sentences=20,
):
    file_list = file_list_folders(input_dir_path, ".xml")

    print("Selecting Patients")
    all_subject_ids = []
    for file in tqdm(file_list):
        filename = os.path.splitext(file)[0]
        subject_id = os.path.basename(filename)[0:3]
        all_subject_ids.append(subject_id)
    sub_ids_list = list(set(all_subject_ids))
    random.shuffle(sub_ids_list)
    print(len(sub_ids_list))
    selected_subject_ids = sub_ids_list[:number_of_patients]
    assert len(selected_subject_ids) == number_of_patients

    print("Preparing Sentences")
    all_sentences = {}
    for file in tqdm(file_list):
        filename = os.path.splitext(file)[0]
        subject_id = os.path.basename(filename)[0:3]
        if subject_id in selected_subject_ids:
            with open(file, "r") as f:
                data = f.read()
            Bs_data = BeautifulSoup(data, "xml")
            b_text = Bs_data.find_all("TEXT")
            file_sentences = []
            if b_text:
                for text in b_text:
                    list_of_sentences = nltk.sent_tokenize(text.string)
                    list_of_sentences = list(set(list_of_sentences))
                    file_sentences.append(list_of_sentences)
            if file_sentences:
                if subject_id not in all_sentences.keys():
                    all_sentences[subject_id] = file_sentences
                else:
                    all_sentences[subject_id].extend(file_sentences)

    # get sentences for selected patients
    selected_subject_texts = [
        [item for sublist in v for item in sublist]
        for k, v in all_sentences.items()
        if k in selected_subject_ids
    ]
    selected_subject_texts = [
        item for sublist in selected_subject_texts for item in sublist
    ]
    print(len(selected_subject_texts))

    # select sentences
    long_sentences = [x for x in selected_subject_texts if len(x) >= min_len_sentences]
    unique_sentences = list(set(long_sentences))
    random.shuffle(unique_sentences)
    selected_sentence_for_MI = unique_sentences[:number_of_sentences]
    assert len(selected_sentence_for_MI) == number_of_sentences
    return selected_sentence_for_MI


def return_lr(test_inputs, model_path, device, sequence_score_type):
    batched_test_inputs = [x for x in create_batch(test_inputs, 1)]

    print("Calculating Sequence Scores Per Batch")
    PLLwordl2r_model = []
    for batch in tqdm(batched_test_inputs):
        target_model = scorer.MaskedLMScorer(model_path, device=str(device))
        target_model_batch = target_model.sequence_score(
            batch, PLL_metric=sequence_score_type, reduction=lambda x: x.mean(0).item()
        )
        torch.cuda.empty_cache()
        PLLwordl2r_model.extend(target_model_batch)

    return PLLwordl2r_model


def return_loss(test_inputs, model_path, device):
    print("Calculating Loss Per Sequence")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    target_model = RobertaForMaskedLM.from_pretrained(model_path)
    losses = []
    for sentence in tqdm(test_inputs):
        input = tokenizer(
            sentence,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = input["input_ids"]
        labels = copy.deepcopy(input_ids)
        attention_mask = input["attention_mask"]
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
        target_model.to(device)
        target_model.eval()
        output = target_model(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask
        )
        # logits = output.logits
        loss = output.loss.item()
        # loss2 = F.cross_entropy(logits.view(
        #     -1,
        #     tokenizer.vocab_size
        # ), labels.view(-1)).item()
        # assert loss == loss2
        losses.append(loss)

    return losses


def plot_score_histograms(data, var1, var2, fig_name, bins=50, loss=True):
    # Plot histograms
    data[var1].hist(alpha=0.5, bins=bins, label=var1)
    data[var2].hist(alpha=0.5, bins=bins, label=var2)

    # Customize plot
    plt.legend()
    if loss:
        plt.title(f"Loss Histogram of {var1} and {var2}")
        plt.xlabel("Loss")
        plt.ylabel("Proportion")
    else:
        plt.title(f"Likelihood Ratio Histogram of {var1} and {var2}")
        plt.xlabel("Likelihood Ratio")
        plt.ylabel("Proportion")

    plt.savefig(f"./figures/all_MI_scores/{fig_name}.png")

    # Show plot
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()


def calculate_test_stats(
    target_results: dict,
    ref_results: dict,
    plot: True,
    save_path: Path,
    experiment: int,
):
    target_name = target_results["model_name"]
    ref_name = ref_results["model_name"]

    # members
    mem_lr_test_statistics = np.subtract(
        target_results["members_lr"], ref_results["members_lr"]
    )
    # non-members
    nonmem_lr_test_statistics = np.subtract(
        target_results["non_members_lr"], ref_results["non_members_lr"]
    )
    # externals
    ext_lr_test_statistics = np.subtract(
        target_results["externals_lr"], ref_results["externals_lr"]
    )
    test_stats_results_dict = dict(
        target=target_name,
        ref=ref_name,
        mem_lr_test_statistics=mem_lr_test_statistics.tolist(),
        nonmem_lr_test_statistics=nonmem_lr_test_statistics.tolist(),
        ext_lr_test_statistics=ext_lr_test_statistics.tolist(),
        mem_loss=target_results["members_loss"],
        non_mem_loss=target_results["non_members_loss"],
        external_loss=target_results["externals_loss"],
    )

    if plot:
        # members vs nonmembers lr
        fig_name = f"lr_non-member_exp={experiment}"
        data_to_plot_nonmems = pd.DataFrame(
            {
                "Members": mem_lr_test_statistics,
                "Non-members": nonmem_lr_test_statistics,
            }
        )
        plot_score_histograms(
            data_to_plot_nonmems,
            var1="Members",
            var2="Non-members",
            fig_name=fig_name,
            bins=100,
            loss=False,
        )
        # members vs externals lr
        fig_name = f"lr_external_exp={experiment}"
        data_to_plot_nonmems = pd.DataFrame(
            {"Members": mem_lr_test_statistics, "Externals": ext_lr_test_statistics}
        )
        plot_score_histograms(
            data_to_plot_nonmems,
            var1="Members",
            var2="Externals",
            fig_name=fig_name,
            bins=100,
            loss=False,
        )

        # members vs non-members loss of target model
        fig_name = f"loss_non_member_exp={experiment}"
        data_to_plot_nonmems = pd.DataFrame(
            {
                "Members": target_results["members_loss"],
                "Non-members": target_results["non_members_loss"],
            }
        )
        plot_score_histograms(
            data_to_plot_nonmems,
            var1="Members",
            var2="Non-members",
            fig_name=fig_name,
            bins=100,
        )

        # members vs externals loss of target model
        fig_name = f"loss_external_exp={experiment}"
        data_to_plot_nonmems = pd.DataFrame(
            {
                "Members": target_results["members_loss"],
                "Externals": target_results["externals_loss"],
            }
        )
        plot_score_histograms(
            data_to_plot_nonmems,
            var1="Members",
            var2="Externals",
            fig_name=fig_name,
            bins=100,
        )

    if os.path.isfile(save_path):
        results_list_so_far = list(read_jsonl(save_path))
        results_list_so_far.append(test_stats_results_dict)
        write_jsonl(file_path=save_path, lines=results_list_so_far)
    else:
        results_list_so_far = [test_stats_results_dict]
        write_jsonl(file_path=save_path, lines=results_list_so_far)

    return test_stats_results_dict


def return_predictions_labels(
    member_test_statistic, nonmem_test_statistic, ext_test_statistic, alpha
):
    mem_test_statistic = sorted(member_test_statistic)
    mem_test_statistic_arr = np.array(mem_test_statistic)
    mem_alpha = np.quantile(mem_test_statistic_arr, alpha)
    # mem
    mem_predictions = [1 if x > mem_alpha else 0 for x in mem_test_statistic]
    mem_labels = [1] * len(nonmem_test_statistic)
    # nonmem
    # nonmem_predictions_holder = [1] * len(nonmem_test_statistic)
    nonmem_predictions = [0 if x < mem_alpha else 1 for x in nonmem_test_statistic]
    nonmem_labels = [0] * len(nonmem_test_statistic)
    # ext
    # nonmem
    ext_predictions = [0 if x < mem_alpha else 1 for x in ext_test_statistic]
    ext_labels = [0] * len(nonmem_test_statistic)
    return (
        mem_predictions,
        mem_labels,
        nonmem_predictions,
        nonmem_labels,
        ext_predictions,
        ext_labels,
    )


def return_auc_roc_over_alphas(
    member_test_statistic, nonmem_test_statistic, ext_test_statistic, alphas
):
    mem_labels = [1] * len(nonmem_test_statistic)
    nonmem_labels = [0] * len(nonmem_test_statistic)
    ext_labels = [0] * len(nonmem_test_statistic)

    nonmem_mem_labels = mem_labels + nonmem_labels
    ext_mem_labels = mem_labels + ext_labels
    all_nonmem_mem_X = []
    all_ext_mem_X = []

    for alpha in alphas:
        mem_test_statistic = sorted(member_test_statistic)
        mem_test_statistic_arr = np.array(mem_test_statistic)
        mem_alpha = np.quantile(mem_test_statistic_arr, alpha)
        # mem
        mem_predictions = [1 if x > mem_alpha else 0 for x in mem_test_statistic]
        # calculate the probability per example
        mem_chance_of_pos = (mem_predictions.count(1)) / len(mem_predictions)
        mem_chance_of_neg = 1 - mem_chance_of_pos
        mem_X = [
            mem_chance_of_pos if x == 1 else mem_chance_of_neg for x in mem_predictions
        ]

        # nonmem
        # nonmem_predictions_holder = [1] * len(nonmem_test_statistic)
        nonmem_predictions = [0 if x < mem_alpha else 1 for x in nonmem_test_statistic]
        # calculate the probability per example
        nonmem_chance_of_pos = (nonmem_predictions.count(1)) / len(nonmem_predictions)
        nonmem_chance_of_neg = 1 - nonmem_chance_of_pos
        nonmem_X = [
            nonmem_chance_of_pos if x == 1 else nonmem_chance_of_neg
            for x in nonmem_predictions
        ]
        nonmem_mem_X = mem_X + nonmem_X
        all_nonmem_mem_X.append(nonmem_mem_X)

        # ext
        # nonmem
        ext_predictions = [0 if x < mem_alpha else 1 for x in ext_test_statistic]
        # calculate the probability per example
        ext_chance_of_pos = (ext_predictions.count(1)) / len(ext_predictions)
        ext_chance_of_neg = 1 - ext_chance_of_pos
        ext_X = [
            ext_chance_of_pos if x == 1 else ext_chance_of_neg for x in ext_predictions
        ]
        ext_mem_X = mem_X + ext_X
        all_ext_mem_X.append(ext_mem_X)

    all_nonmem_mem_X = np.array(all_nonmem_mem_X).T
    all_ext_mem_X = np.array(all_ext_mem_X).T
    nonmem_mem_labels = np.array(nonmem_mem_labels)
    ext_mem_labels = np.array(ext_mem_labels)

    nonmem_clf = LogisticRegression(solver="liblinear", random_state=0).fit(
        all_nonmem_mem_X, nonmem_mem_labels
    )
    nonmen_roc_auc_score = roc_auc_score(
        nonmem_mem_labels, nonmem_clf.predict_proba(all_nonmem_mem_X)[:, 1]
    )

    ext_clf = LogisticRegression(solver="liblinear", random_state=0).fit(
        all_ext_mem_X, ext_mem_labels
    )
    ext_roc_auc_score = roc_auc_score(
        ext_mem_labels, ext_clf.predict_proba(all_ext_mem_X)[:, 1]
    )

    return round(nonmen_roc_auc_score, 3), round(ext_roc_auc_score, 3)


def calculate_results(loss_true, loss_pred, lr_true, lr_pred):
    # loss
    loss_true = np.array(loss_true)
    loss_pred = np.array(loss_pred)
    lr_true = np.array(lr_true)
    lr_pred = np.array(lr_pred)
    loss_precision = round(
        metrics.precision_score(y_true=loss_true, y_pred=loss_pred, pos_label=1), 3
    )
    loss_recall = round(
        metrics.recall_score(y_true=loss_true, y_pred=loss_pred, pos_label=1), 3
    )
    loss_f1 = round(
        metrics.f1_score(y_true=loss_true, y_pred=loss_pred, pos_label=1), 3
    )
    # lr
    lr_precision = round(
        metrics.precision_score(y_true=lr_true, y_pred=lr_pred, pos_label=1), 3
    )
    lr_recall = round(
        metrics.recall_score(y_true=lr_true, y_pred=lr_pred, pos_label=1), 3
    )
    lr_f1 = round(metrics.f1_score(y_true=lr_true, y_pred=lr_pred, pos_label=1), 3)

    return dict(
        loss_precision=loss_precision,
        loss_recall=loss_recall,
        loss_f1=loss_f1,
        lr_precision=lr_precision,
        lr_recall=lr_recall,
        lr_f1=lr_f1,
    )
