#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim)
# Title: preprocess.py
# Explanation: Loads the csv file of the human data and preprocess it for the simulation.

import collections
import itertools
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def prepare_data(file_path: str, participant_id: str = None, participant_index: int = None,
                 load_full_df: bool = False) -> Dict:
    try:
        df = pd.read_csv(file_path, index_col=0)
    except FileNotFoundError:
        print(f"FATAL ERROR: Data file not found at '{file_path}'")
        return {}

    all_pids = df['prolific_id'].unique()
    rt_exec_col = 'ThreeRespRT'
    participant_stats = df.groupby('prolific_id').agg(
        accuracy=('accuracy', 'mean'),
        n_trials=('accuracy', 'count'),
        mean_rt_exec=(rt_exec_col, 'mean')
    )
    pids_to_exclude = set()
    for pid, data in participant_stats.iterrows():
        n_trials = int(data['n_trials'])
        n_correct = int(data['accuracy'] * n_trials)
        result = binomtest(k=n_correct, n=n_trials, p=0.1, alternative='greater')
        if result.pvalue > 0.05:
            pids_to_exclude.add(pid)
        if data['accuracy'] >= 0.90:
            pids_to_exclude.add(pid)

    fast_pids = participant_stats[participant_stats['mean_rt_exec'] < 500].index
    pids_to_exclude.update(fast_pids)
    mean_of_means = participant_stats['mean_rt_exec'].mean()
    sd_of_means = participant_stats['mean_rt_exec'].std()
    slow_cutoff = mean_of_means + (3 * sd_of_means)
    slow_pids = participant_stats[participant_stats['mean_rt_exec'] > slow_cutoff].index
    pids_to_exclude.update(slow_pids)
    valid_prolific_ids = [pid for pid in all_pids if pid not in pids_to_exclude]

    if not valid_prolific_ids:
        raise ValueError("No subjects remaining after applying exclusion criteria.")

    if not load_full_df:
        return {"valid_prolific_ids": valid_prolific_ids}

    selected_pid = None
    if participant_id:
        if participant_id in df['prolific_id'].values:
            selected_pid = participant_id
        else:
            return {}
    elif participant_index is not None:
        if participant_index < len(valid_prolific_ids):
            selected_pid = valid_prolific_ids[participant_index]
        else:
            print(f"FATAL ERROR: participant_index {participant_index} is out of bounds for the cleaned list. "
                  f"There are only {len(valid_prolific_ids)} valid participants.")
            return {}

    if not selected_pid:
        print(f"Could not find participant.")
        return {}

    subj_df_raw = df[df['prolific_id'] == selected_pid].reset_index(drop=True)
    valid_trials_mask = subj_df_raw['problem_solving'].notna()
    subj_df = subj_df_raw[valid_trials_mask].reset_index(drop=True)

    choice_cols = ['choice_color', 'choice_fill', 'choice_shape', 'choice_back']
    subj_df.dropna(subset=choice_cols, inplace=True)
    subj_df.reset_index(drop=True, inplace=True)

    choice_cols = ['choice_color', 'choice_fill', 'choice_shape', 'choice_back']
    if subj_df[choice_cols].isnull().values.any():
        # print("\n" + "=" * 20 + " DEBUGGING REPORT " + "=" * 20)
        # print(f"!!! Found NaN values in the choice data for participant: {selected_pid}")
        nan_rows = subj_df[subj_df[choice_cols].isnull().any(axis=1)]
        # print("The following trials have incomplete choice data:")
        # print(nan_rows[['trial_num'] + choice_cols].to_string())
        # print("=" * 58 + "\n")

    if subj_df.empty:
        return {}

    num_trials = len(subj_df)
    attributes = {
        "C": ["red", "yellow", "green"], "F": ["empty", "half", "full"],
        "S": ["circle", "triangle", "square"], "B": ["black", "grey", "white"]
    }
    attribute_names = list(attributes.keys())
    card_cols = ['one', 'two', 'three', 'four', 'five']

    digit_cards = np.zeros((num_trials, 5, 4), dtype=int)
    for i in range(num_trials):
        for card_idx, card_col_name in enumerate(card_cols):
            card_str_features = subj_df.loc[i, card_col_name].rstrip(".png").split("_")
            for att_idx, att_name in enumerate(attribute_names):
                feature_str = card_str_features[att_idx]
                digit_cards[i, card_idx, att_idx] = attributes[att_name].index(feature_str)

    combinations_of_3 = list(itertools.combinations(range(5), r=3))
    relation_matrices = np.zeros((num_trials, 10, 4), dtype=int)
    for i in range(num_trials):
        for att_idx in range(len(attribute_names)):
            for combo_idx, combo in enumerate(combinations_of_3):
                three_features = digit_cards[i, list(combo), att_idx]
                relation_matrices[i, combo_idx, att_idx] = len(np.unique(three_features))

    projection_matrices = np.zeros((num_trials, 4, 4))
    card_availability = []

    for i in range(num_trials):
        trial_rel_matrix = relation_matrices[i, :, :]
        atts_for_this_trial = []
        for att_idx in range(len(attribute_names)):
            rels_for_att = trial_rel_matrix[:, att_idx]
            bins = {1: [], 2: [], 3: []}
            for choice_idx, rel in enumerate(rels_for_att):
                bins[rel].append(choice_idx)
            atts_for_this_trial.append([bins[1], bins[2], bins[3]])
            projection_matrices[i, att_idx, 1] = len(bins[1])
            projection_matrices[i, att_idx, 2] = len(bins[2])
            projection_matrices[i, att_idx, 3] = len(bins[3])
        projection_matrices[i, :, 0] = np.sum(projection_matrices[i, :, 1:], axis=1)
        card_availability.append(atts_for_this_trial)

    card_name_combos = sorted([''.join(c) for c in itertools.combinations('12345', 3)])

    true_answers = np.array([
        card_name_combos.index(str(int(ans))) for ans in subj_df['correct_response']
    ])

    num_candidates_per_attr = np.zeros((num_trials, 4, 3), dtype=int)
    for i in range(num_trials):
        for att_idx in range(len(attribute_names)):
            cnts = collections.Counter(relation_matrices[i, :, att_idx])
            num_candidates_per_attr[i, att_idx, 0] = cnts.get(1, 0)
            num_candidates_per_attr[i, att_idx, 1] = cnts.get(2, 0)
            num_candidates_per_attr[i, att_idx, 2] = cnts.get(3, 0)

    choice_vectors = subj_df[['choice_color', 'choice_fill', 'choice_shape', 'choice_back']].to_numpy(dtype=int)
    true_vectors = subj_df[['modSum_color', 'modSum_fill', 'modSum_shape', 'modSum_back']].to_numpy(dtype=int)

    return {
        "prolific_id": selected_pid,
        "relation_matrices": relation_matrices,
        "true_answers": true_answers,
        "attributes": attribute_names,
        "num_trials": num_trials,
        "digit_cards": digit_cards,
        "projection_matrices": projection_matrices,
        "card_availability": card_availability,
        "num_candidates_per_attr": num_candidates_per_attr,
        "choice_vectors": choice_vectors,
        "true_vectors": true_vectors,
        "valid_prolific_ids": valid_prolific_ids,
        "solvability": subj_df['solvability'].values,
        "confidence": subj_df['confidence'].values,
        "insight": subj_df['insight'].values,
        "actualDim": subj_df['actualDim'].values,
    }
