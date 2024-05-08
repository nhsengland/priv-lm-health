"""Script to Prepare Clinical Notes Data from MIMIC-III and i2b2 2014_risk_factors for Membership Inference Attacks."""

from pathlib import Path

import typer

from privlm.mi_attack import (
    prep_i2b2_for_MI,
    prep_mimic_for_MI,
    get_subject_ids,
    create_non_member_df,
    create_member_df,
)
from privlm.utils import file_list_folders, write_jsonl


def main(
    mimic_training_dir_path: Path = typer.Option(
        default="../datasets/mimic/training",
        help=("path to  directory containing files used to train the target model."),
    ),
    mimic_nontraining_dir_path: Path = typer.Option(
        default="../datasets/mimic/nontraining",
        help=(
            "path to directory containing files not used in training of target model"
            " but from the same dataset."
        ),
    ),
    external_dir_path: str = typer.Option(
        default="../datasets/i2b2/n2c2_2014_risk_factors",
        help=(
            "path to directory containing files not used in training from an"
            " external dataset of a similar domain to the training data."
        ),
    ),
    number_of_patients: int = typer.Option(
        default=90,
        help=(
            "number of patients to include in the members, non-members and"
            " externals datasets"
        ),
    ),
    number_of_sentences: int = typer.Option(
        default=1000,
        help=(
            "number of sentences from across the patients to include in the members,"
            " non-members and externals datasets"
        ),
    ),
    output_path: str = typer.Option(
        default="../datasets/membership_inf/",
        help="path to dir to save members, non-members and externals datasets to.",
    ),
):
    ############ READ IN RAW DATA ############

    training_files = sorted(file_list_folders(mimic_training_dir_path, ".csv"))
    non_training_files = sorted(file_list_folders(mimic_nontraining_dir_path, ".csv"))

    ############ REMOVE OVERLAPPING PATIENTS BETWEEN MEMBERS & NON-MEMBERS ############

    member_subject_ids = get_subject_ids(training_files)
    non_training_subject_ids = get_subject_ids(non_training_files)
    non_member_subject_ids = [
        x for x in non_training_subject_ids if x not in member_subject_ids
    ]

    non_member_df = create_non_member_df(non_training_files, non_member_subject_ids)
    member_df = create_member_df(training_files)

    ############ SELECT PATIENTS AND SENTENCES ############

    members = prep_mimic_for_MI(
        member_df,
        number_of_patients=number_of_patients,
        number_of_sentences=number_of_sentences,
    )

    non_members = prep_mimic_for_MI(
        non_member_df,
        number_of_patients=number_of_patients,
        number_of_sentences=number_of_sentences,
    )

    externals = prep_i2b2_for_MI(
        external_dir_path,
        number_of_patients=number_of_patients,
        number_of_sentences=number_of_sentences,
    )

    ############ WRITE OUT PROCESSED DATA ############

    write_jsonl(
        file_path=(
            f"{output_path}/externals_{number_of_patients}"
            f"patients_{number_of_sentences}sentences.json"
        ),
        lines=externals,
    )

    write_jsonl(
        file_path=(
            f"{output_path}/members_{number_of_patients}"
            f"patients_{number_of_sentences}sentences.json"
        ),
        lines=members,
    )

    write_jsonl(
        file_path=(
            f"{output_path}/non_members_{number_of_patients}"
            "patients_{number_of_sentences}sentences.json"
        ),
        lines=non_members,
    )


if __name__ == "__main__":
    typer.run(main)
