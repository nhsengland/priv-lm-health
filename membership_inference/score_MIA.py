"""
Script to Run Likelihood Ratio Membership Inference Attack on Masked Language Models
using a Reference Model. Assumes data already preprocessed into member, non-member
and external datasets.
"""

import random
from pathlib import Path

import numpy as np
import torch
import typer

from privlm.mi_attack import (
    return_loss,
    return_lr,
    calculate_test_stats,
    calculate_results,
    return_auc_roc_over_alphas,
    return_predictions_labels,
)
from privlm.utils import write_jsonl, read_jsonl


def main(
    target_model: Path = typer.Option(
        "../pretrained_models/Clinical-RoBERTa-None",
        help=(
            "path to the target model you want to stage the attack against, "
            "must be compatible with huggingface transformers."
        ),
    ),
    reference_model: Path = typer.Option(
        "../pretrained_models/RoBERTa-base-PM-Voc",
        help=(
            "path to the reference model you want to use in the likelihood"
            " ratio attack, must be compatible with huggingface transformers."
        ),
    ),
    member_file_path: Path = typer.Option(
        default="../datasets/membership_inf/members_90patients_1000sentences.json",
        help="path to member dataset file.",
    ),
    non_member_file_path: Path = typer.Option(
        default="../datasets/membership_inf/non_members_90patients_1000sentences.json",
        help="path to non-member dataset file.",
    ),
    external_file_path: str = typer.Option(
        default="../datasets/membership_inf/externals_90patients_1000sentences.json",
        help="path to external dataset file.",
    ),
    mi_test_statistics_file_path: Path = typer.Option(
        default="./results/all_MI_test_statistics.json",
        help="path to file to save raw test statistics to.",
    ),
    mi_results_file_path: Path = typer.Option(
        default="./results/all_MI_privacy_results.json",
        help="path to file to save final privacy quantification results to.",
    ),
):
    ############ SET VARS ############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    random.seed(1)

    ############ IMPORT DATA  ############
    members = list(read_jsonl(member_file_path))
    non_members = list(read_jsonl(non_member_file_path))
    externals = list(read_jsonl(external_file_path))

    ############ CALCULATE LOSS AND LIKELIHOOD RATIOS FOR TARGET & REF MODELS #########

    models = [target_model, reference_model]
    all_model_results = []
    for model in models:
        print(f"Calculating Loss & LR for {model}")
        # calculate loss
        members_loss = return_loss(test_inputs=members, model_path=model, device=device)
        non_members_loss = return_loss(
            test_inputs=non_members, model_path=model, device=device
        )
        externals_loss = return_loss(
            test_inputs=externals, model_path=model, device=device
        )

        # calculate likelihood ratios
        members_lr = return_lr(
            test_inputs=members,
            model_path=model,
            device=device,
            sequence_score_type="within_word_l2r",
        )
        non_members_lr = return_lr(
            test_inputs=non_members,
            model_path=model,
            device=device,
            sequence_score_type="within_word_l2r",
        )
        externals_lr = return_lr(
            test_inputs=externals,
            model_path=model,
            device=device,
            sequence_score_type="within_word_l2r",
        )

        # append results
        results_dict = dict(
            model_name=str(model),
            members_lr=list(members_lr),
            non_members_lr=list(non_members_lr),
            externals_lr=list(externals_lr),
            members_loss=members_loss,
            non_members_loss=non_members_loss,
            externals_loss=externals_loss,
        )
        all_model_results.append(results_dict)

    ############ CARRY OUT MIA ############

    print(f"MIA between {target_model} and {reference_model}")
    raw_target_results = [
        d for d in all_model_results if d["model_name"] == target_model
    ][0]
    raw_ref_results = [
        d for d in all_model_results if d["model_name"] == reference_model
    ][0]
    experiment = 1
    test_stats_results_dict = calculate_test_stats(
        raw_target_results,
        raw_ref_results,
        plot=True,
        save_path=mi_test_statistics_file_path,
        experiment=experiment,
    )

    ############ DEFINE THRESHOLD T ############
    # lr FP rate alpha
    lr_alpha = 0.1

    # loss FP rate alpha
    loss_alpha = 0.1

    ############ CALCULATE PREDICTIONS AND LABELS GIVEN ALPHA ############
    # lr
    (
        member_lr_predictions,
        member_lr_labels,
        non_member_lr_predictions,
        non_member_lr_labels,
        ext_lr_predictions,
        ext_lr_labels,
    ) = return_predictions_labels(
        member_test_statistic=test_stats_results_dict["mem_lr_test_statistics"],
        nonmem_test_statistic=test_stats_results_dict["nonmem_lr_test_statistics"],
        ext_test_statistic=test_stats_results_dict["ext_lr_test_statistics"],
        alpha=lr_alpha,
    )
    mem_vs_nonmem_lr_preds = member_lr_predictions + non_member_lr_predictions
    mem_vs_nonmem_lr_labels = member_lr_labels + non_member_lr_labels
    mem_vs_ext_lr_pred = member_lr_predictions + ext_lr_predictions
    mem_vs_ext_lr_labels = member_lr_labels + ext_lr_labels

    # loss
    (
        member_loss_predictions,
        member_loss_labels,
        non_member_loss_predictions,
        non_member_loss_labels,
        ext_loss_predictions,
        ext_loss_labels,
    ) = return_predictions_labels(
        member_test_statistic=test_stats_results_dict["mem_loss"],
        nonmem_test_statistic=test_stats_results_dict["non_mem_loss"],
        ext_test_statistic=test_stats_results_dict["external_loss"],
        alpha=loss_alpha,
    )

    mem_vs_nonmem_loss_preds = member_loss_predictions + non_member_loss_predictions
    mem_vs_nonmem_loss_labels = member_loss_labels + non_member_loss_labels
    mem_vs_ext_loss_pred = member_loss_predictions + ext_loss_predictions
    mem_vs_ext_loss_labels = member_loss_labels + ext_loss_labels

    ############ QAUNTIFY PRIVACY RISK ############

    alphas = np.arange(0, 1.1, 0.1)

    # lr
    lr_nonmen_roc_auc_score, lr_ext_roc_auc_score = return_auc_roc_over_alphas(
        member_test_statistic=test_stats_results_dict["mem_lr_test_statistics"],
        nonmem_test_statistic=test_stats_results_dict["nonmem_lr_test_statistics"],
        ext_test_statistic=test_stats_results_dict["ext_lr_test_statistics"],
        alphas=alphas,
    )

    # loss
    loss_nonmen_roc_auc_score, loss_ext_roc_auc_score = return_auc_roc_over_alphas(
        member_test_statistic=test_stats_results_dict["mem_loss"],
        nonmem_test_statistic=test_stats_results_dict["non_mem_loss"],
        ext_test_statistic=test_stats_results_dict["external_loss"],
        alphas=alphas,
    )

    # members vs non_members
    non_member_results = calculate_results(
        loss_true=mem_vs_nonmem_loss_labels,
        loss_pred=mem_vs_nonmem_loss_preds,
        lr_true=mem_vs_nonmem_lr_labels,
        lr_pred=mem_vs_nonmem_lr_preds,
    )
    non_member_results["lr_auc"] = lr_nonmen_roc_auc_score
    non_member_results["loss_auc"] = loss_nonmen_roc_auc_score

    # members vs externals
    external_results = calculate_results(
        loss_true=mem_vs_ext_loss_labels,
        loss_pred=mem_vs_ext_loss_pred,
        lr_true=mem_vs_ext_lr_labels,
        lr_pred=mem_vs_ext_lr_pred,
    )
    external_results["lr_auc"] = lr_ext_roc_auc_score
    external_results["loss_auc"] = loss_ext_roc_auc_score

    results_dict = dict(
        target=models["target"],
        ref=models["ref"],
        non_member_results=non_member_results,
        external_results=external_results,
    )

    print(results_dict)

    ############ WRITE OUT RESULTS ############

    write_jsonl(file_path=mi_results_file_path, lines=[results_dict])


if __name__ == "__main__":
    typer.run(main)
