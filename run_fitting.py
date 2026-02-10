#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim)
# Title: run_fitting.py
# Explanation: Implements the two-stage fitting pipeline (mcts_only, metacontroller_fit) using Bayesian optimization.
#              Defines objective functions (log-likelihood) to find model parameters that best explain human choice.

import argparse
import os
import sys
import time
import traceback
import warnings
from functools import partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from agents import AgentMCTS
from environment import Environment
from metacontroller import Metacontroller
from preprocess import prepare_data

warnings.filterwarnings("ignore", category=UserWarning)

MCTS_ONLY_SPACE = [
    Real(0.01, 1, name='lr_same', prior='log-uniform'),
    Real(0.01, 1, name='lr_diff', prior='log-uniform'),
    Real(0.01, 1, name='lr_pair', prior='log-uniform'),
    Real(0.01, 1, name='lr_attribute', prior='log-uniform'),
    Real(0.2, 2.0, name='selection_temperature', prior='uniform'),

    Real(0.0, 1.0, name='clarity_threshold', prior='uniform'),
    Real(1.0, 20.0, name='sigmoid_slope', prior='log-uniform'),
    Real(1.0, 50.0, name='clarity_weight', prior='log-uniform'),

    Integer(20, 500, name='mcts_n_simulations', prior='log-uniform'),
    Integer(5, 400, name='min_sims', prior='uniform'),
    Real(0.1, 8.0, name='ucb_exploration_constant', prior='log-uniform'),
    Real(0.01, 0.2, name='learning_noise', prior='uniform'),
]

EVC_SPACE = [
    Real(0.001, 10.0, name='cost_mcts', prior='log-uniform'),
    Real(0.001, 10.0, name='meta_temperature', prior='log-uniform'),
]

HYBRID_SPACE = [
    Real(0.001, 10.0, name='cost_mcts', prior='log-uniform'),
    Real(0.01, 80.0, name='hybrid_mf_scaler', prior='log-uniform'),
    Real(0.001, 10.0, name='meta_temperature', prior='log-uniform'),
    Real(0.01, 10.0, name='confidence_scaler', prior='uniform'),
    # The "Standard of Proof" for the Latch
    Real(0.5, 0.99, name='insight_threshold', prior='uniform'),
]

INSIGHT_SPACE = [
    Real(0.1, 50.0, name='insight_slope', prior='log-uniform'),
    Real(0.0, 5.0, name='insight_threshold', prior='uniform'),
    Real(0.01, 5.0, name='meta_temperature', prior='log-uniform'),
]

VALUE_GAP_SPACE = [
    Real(0.1, 100.0, name='gap_slope', prior='log-uniform'),
    Real(-2.0, 2.0, name='gap_intercept', prior='uniform'),
    Real(0.01, 5.0, name='meta_temperature', prior='log-uniform'),
]

BELIEF_GAP_SPACE = [
    Real(1.0, 200.0, name='gap_slope', prior='log-uniform'),
    Real(-0.5, 0.5, name='gap_intercept', prior='uniform'),
    Real(0.01, 5.0, name='meta_temperature', prior='log-uniform'),
]


def calculate_mcts_log_likelihood(params: dict, participant_data: dict, group_means: pd.Series, group_stds: pd.Series,
                                  lambda_strength: float) -> float:
    pid = participant_data.get('prolific_id', 'UnknownPID')
    invalid_env_count = 0
    zero_prob_count = 0
    missing_choice_idx_count = 0
    try:
        agent = AgentMCTS(attributes=participant_data['attributes'], **params)
        total_log_likelihood = 0.0
        num_valid_trials_for_ll = 0

        for i in range(len(participant_data['true_answers'])):
            env = Environment(participant_data, i, participant_data["true_answers"][i], attributes=agent.attributes)
            if not env.is_valid():
                invalid_env_count += 1
                continue

            # H-scores for ALL 10 options based on CURRENT agent state
            all_scores = np.zeros(10, dtype=np.float64)
            for opt_idx in range(10):
                relations = env.get_triplet_relations(opt_idx)
                if relations is not None:
                    # Use the agent's current v_relation table to score all options
                    all_scores[opt_idx] = agent.calculate_learned_h_score(relations)
                else:
                    all_scores[opt_idx] = -np.inf  # Penalize invalid/missing options

            # Apply softmax over ALL 10 scores
            effective_temp = max(agent.selection_temperature, 1e-9)
            stable_scores = all_scores - np.nanmax(all_scores)  # nanmax handles -inf
            stable_scores[np.isinf(stable_scores)] = -1e9  # Replace -inf
            exp_scores = np.exp(stable_scores / effective_temp)
            sum_exp_scores = np.nansum(exp_scores)

            if sum_exp_scores > 1e-9 and not np.isnan(sum_exp_scores):
                final_policy = exp_scores / sum_exp_scores
                final_policy = np.nan_to_num(final_policy, nan=0.0)
                policy_sum = final_policy.sum()
                if policy_sum > 1e-9:
                    final_policy /= policy_sum
                else:
                    final_policy = np.ones(10) / 10.0
            else:
                final_policy = np.ones(10) / 10.0

            human_vector = participant_data['choice_vectors'][i]
            human_choice_idx = -1
            if 'relation_matrices' in participant_data and i < len(participant_data['relation_matrices']):
                relation_matrix_trial_i = participant_data['relation_matrices'][i]
                if relation_matrix_trial_i is not None and isinstance(relation_matrix_trial_i, np.ndarray):
                    try:
                        human_vector_np = np.array(human_vector)
                        matching_indices = np.where(np.all(relation_matrix_trial_i == human_vector_np, axis=1))[0]
                        if matching_indices.size > 0:
                            human_choice_idx = matching_indices[0]
                        else:
                            missing_choice_idx_count += 1
                    except (ValueError, TypeError) as e:
                        missing_choice_idx_count += 1
                else:
                    missing_choice_idx_count += 1
            else:
                missing_choice_idx_count += 1

            if human_choice_idx != -1:
                prob = final_policy[human_choice_idx]
                if prob > 1e-9:
                    total_log_likelihood += np.log(prob)
                else:
                    total_log_likelihood += -20  # Apply penalty
                    zero_prob_count += 1
                num_valid_trials_for_ll += 1

            # Simulate agent trial and learning (to update state for the next trial)
            if i < 30:  # reinforcement phase
                try:
                    # Unpack 5 items and discard the 5th (Fusion Score)
                    max_sims = params['mcts_n_simulations']
                    agent.mcts_search.n_simulations = max_sims

                    _, search_path, sim_policy, _, _ = agent.solve_one_trial(env)

                    if sim_policy is None or len(sim_policy) != 10 or np.isclose(sim_policy.sum(), 0):
                        sim_policy = np.ones(10) / 10.0
                except Exception as solve_err:
                    print(
                        f"ERROR (PID {pid}, Trial {i}): agent.solve_one_trial for learning failed: {solve_err}. Params: {params}")
                    continue  # Skip learning if sim fails

                agent_choice_idx = np.random.choice(10, p=sim_policy)
                agent_reward = 1.0 if agent_choice_idx == env.true_set_index else 0.0
                agent_choice_vector = env.get_triplet_relations(agent_choice_idx)
                if agent_choice_vector is not None:
                    agent.learn_from_trial_outcome(agent_choice_vector, agent_reward)

                if search_path:
                    for step in search_path:
                        if isinstance(step, dict) and 'attribute_chosen' in step and 'reward' in step:
                            agent.learn_from_search_strategy(step['attribute_chosen'], step['reward'])

        if invalid_env_count > 0: pass
        if zero_prob_count > 0: pass
        if missing_choice_idx_count > 0: pass
        if num_valid_trials_for_ll == 0:
            return -1e9

        # Regularization penalty (no use)
        penalty = 0.0
        if group_means is not None and group_stds is not None and lambda_strength > 0:
            for param_name, value in params.items():
                if param_name in group_means.index and param_name in group_stds.index and group_stds[param_name] > 0:
                    penalty += ((value - group_means[param_name]) ** 2) / (2 * group_stds[param_name] ** 2)

        final_ll = total_log_likelihood - (lambda_strength * penalty)
        if np.isnan(final_ll) or np.isinf(final_ll): print(
            f"ERROR (PID {pid}): Final LL is NaN or Inf. Params: {params}"); return -1e9
        return final_ll

    except Exception as e:
        print(f"ERROR during MCTS LL calculation for PID {pid}. Params: {params}. Error: {e}")
        traceback.print_exc()
        return -1e9


def calculate_metacontroller_log_likelihood(params: dict, participant_data: dict, fixed_agent_params: dict,
                                            meta_model_type: str,
                                            group_means: pd.Series, group_stds: pd.Series,
                                            lambda_strength: float) -> float:
    pid = participant_data.get('prolific_id', 'UnknownPID')
    invalid_env_count = 0
    zero_prob_count = 0
    missing_choice_idx_count = 0
    full_params = {}
    try:
        fixed_agent_params['clarity_threshold'] = fixed_agent_params.get('clarity_threshold', 0.5)
        fixed_agent_params['sigmoid_slope'] = fixed_agent_params.get('sigmoid_slope', 5.0)
        fixed_agent_params['clarity_weight'] = fixed_agent_params.get('clarity_weight', 1.0)

        full_params = {**fixed_agent_params, **params,
                       'attributes': participant_data['attributes'],
                       'meta_model_type': meta_model_type,
                       'use_internal_teacher': False}

        agent = Metacontroller(**full_params)
        total_log_likelihood = 0.0
        num_valid_trials_for_ll = 0

        for i in range(len(participant_data['true_answers'])):
            env = Environment(participant_data, i, participant_data["true_answers"][i],
                              attributes=agent.mb_agent.attributes)

            if not env.is_valid():
                invalid_env_count += 1
                continue

            try:
                chosen_agent_str, prob_mb, dynamic_n_sims = agent.choose_agent(env)
            except Exception as choose_err:
                print(f"ERROR (PID {pid}, Trial {i}): agent.choose_agent failed: {choose_err}. Params: {full_params}")
                total_log_likelihood += -50
                continue

            # if MB is active
            all_scores_mb = np.zeros(10, dtype=np.float64)
            for opt_idx in range(10):
                relations = env.get_triplet_relations(opt_idx)
                if relations is not None:
                    all_scores_mb[opt_idx] = agent.mb_agent.calculate_learned_h_score(relations)
                else:
                    all_scores_mb[opt_idx] = -np.inf

            effective_temp_mb = max(agent.mb_agent.selection_temperature, 1e-9)
            stable_scores_mb = all_scores_mb - np.nanmax(all_scores_mb)
            stable_scores_mb[np.isinf(stable_scores_mb)] = -1e9
            exp_scores_mb = np.exp(stable_scores_mb / effective_temp_mb)
            sum_exp_scores_mb = np.nansum(exp_scores_mb)

            if sum_exp_scores_mb > 1e-9 and not np.isnan(sum_exp_scores_mb):
                final_policy_mb = exp_scores_mb / sum_exp_scores_mb
            else:
                final_policy_mb = np.ones(10) / 10.0

            # if MF is active
            all_scores_mf = np.zeros(10, dtype=np.float64)
            for opt_idx in range(10):
                relations = env.get_triplet_relations(opt_idx)
                if relations is not None:
                    all_scores_mf[opt_idx] = agent.mf_agent.calculate_learned_h_score(relations)
                else:
                    all_scores_mf[opt_idx] = -np.inf

            effective_temp_mf = max(agent.mf_agent.selection_temperature, 1e-9)
            stable_scores_mf = all_scores_mf - np.nanmax(all_scores_mf)
            stable_scores_mf[np.isinf(stable_scores_mf)] = -1e9
            exp_scores_mf = np.exp(stable_scores_mf / effective_temp_mf)
            sum_exp_scores_mf = np.nansum(exp_scores_mf)
            if sum_exp_scores_mf > 1e-9 and not np.isnan(sum_exp_scores_mf):
                final_policy_mf = exp_scores_mf / sum_exp_scores_mf
            else:
                final_policy_mf = np.ones(10) / 10.0

            human_vector = participant_data['choice_vectors'][i]
            human_choice_idx = -1
            if 'relation_matrices' in participant_data and i < len(participant_data['relation_matrices']):
                relation_matrix_trial_i = participant_data['relation_matrices'][i]
                if relation_matrix_trial_i is not None and isinstance(relation_matrix_trial_i, np.ndarray):
                    try:
                        human_vector_np = np.array(human_vector)
                        matching_indices = np.where(np.all(relation_matrix_trial_i == human_vector_np, axis=1))[0]
                        if matching_indices.size > 0:
                            human_choice_idx = matching_indices[0]
                        else:
                            missing_choice_idx_count += 1
                    except (ValueError, TypeError) as e:
                        missing_choice_idx_count += 1
                else:
                    missing_choice_idx_count += 1
            else:
                missing_choice_idx_count += 1

            if human_choice_idx != -1:
                prob_if_mb = final_policy_mb[human_choice_idx]
                prob_if_mf = final_policy_mf[human_choice_idx]
                marginal_prob = (prob_mb * prob_if_mb) + ((1 - prob_mb) * prob_if_mf)

                if marginal_prob > 1e-9:
                    total_log_likelihood += np.log(marginal_prob)
                else:
                    total_log_likelihood += -20
                    zero_prob_count += 1
                num_valid_trials_for_ll += 1

            final_candidates = set()
            search_path = []

            try:
                chosen_agent_for_learning = 'MB' if np.random.rand() < prob_mb else 'MF'

                if chosen_agent_for_learning == 'MB':
                    # MB returns 5 items including raw_fusion_score
                    final_candidates, search_path, policy_for_learning, _, _ = agent.mb_agent.solve_one_trial(env, dynamic_n_sims=dynamic_n_sims)
                else:
                    # MF
                    final_candidates, search_path, policy_for_learning, _ = agent.mf_agent.solve_one_trial(env)

                if policy_for_learning is None or len(policy_for_learning) != 10 or np.isclose(
                        policy_for_learning.sum(), 0):
                    policy_for_learning = np.ones(10) / 10.0

                agent_choice_idx = np.random.choice(10, p=policy_for_learning)
                accuracy = 1.0 if agent_choice_idx == env.true_set_index else 0.0
                valid_search_path = []
                if search_path: valid_search_path = [step for step in search_path if isinstance(step,
                                                                                                dict) and 'attribute_chosen' in step and 'reward' in step]

                # Latch check inside Metacontroller
                agent.learn_from_trial(env, accuracy, valid_search_path, final_candidates, i,
                                       agent_choice_idx, chosen_agent_for_learning)

            except Exception as learn_err:
                print(f"ERROR (PID {pid}, Trial {i}): Error during agent learning step: {learn_err}. Params: {full_params}")

        if num_valid_trials_for_ll == 0:
            return -1e8

        penalty = 0.0
        if group_means is not None and group_stds is not None and lambda_strength > 0:
            for param_name, value in params.items():
                if param_name in group_means.index and param_name in group_stds.index and group_stds[param_name] > 0:
                    penalty += ((value - group_means[param_name]) ** 2) / (2 * group_stds[param_name] ** 2)

        final_ll = total_log_likelihood - (lambda_strength * penalty)

        if np.isnan(final_ll) or np.isinf(final_ll):
            print(f"ERROR (PID {pid}, Model {meta_model_type}): Final LL is NaN or Inf. Params: {full_params}")
            return -1e8
        return final_ll

    except Exception as e:
        params_to_print_on_error = full_params if full_params else {**fixed_agent_params, **params}
        print(f"ERROR during Metacontroller LL calculation for PID {pid}. Model {meta_model_type}. Params: {params_to_print_on_error}. Error: {e}")
        traceback.print_exc()
        return -1e8


def fit_participant(participant_index: int, participant_id: str, file_path: str, n_calls: int, mode: str,
                    space: list, mcts_priors_df: pd.DataFrame,
                    group_means: pd.Series, group_stds: pd.Series,
                    lambda_strength: float, meta_model_type: str = None):
    start_time = time.time()
    print(f"Starting fitting for P{participant_index + 1} in mode '{mode}' (model: {meta_model_type or 'mcts'})...")
    participant_data = prepare_data(file_path=file_path, participant_index=participant_index, load_full_df=True)
    if not participant_data:
        return {'prolific_id': f"index_{participant_index}", 'error': 'Data preparation failed'}

    if mode == 'mcts_only':
        objective_func = partial(calculate_mcts_log_likelihood, participant_data=participant_data,
                                 group_means=group_means, group_stds=group_stds, lambda_strength=lambda_strength)
    else:
        fixed_params = mcts_priors_df[mcts_priors_df['prolific_id'] == participant_id].iloc[0].to_dict()

        # fixed_params dictionary does NOT contain strategic_influence, fixed at 32.16 in agents.py.
        if 'strategic_influence' in fixed_params: del fixed_params['strategic_influence']
        objective_func = partial(calculate_metacontroller_log_likelihood, participant_data=participant_data,
                                 fixed_agent_params=fixed_params, meta_model_type=meta_model_type,
                                 group_means=group_means, group_stds=group_stds, lambda_strength=lambda_strength)

    @use_named_args(space)
    def objective(**params):
        return -objective_func(params)

    res_gp = gp_minimize(func=objective, dimensions=space, n_calls=n_calls, n_initial_points=20,
                         random_state=int(start_time))
    best_params = {s.name: val for s, val in zip(space, res_gp.x)}
    best_params['log_likelihood'] = -res_gp.fun
    best_params['prolific_id'] = participant_id
    print(f"Finished P{participant_index + 1} in {time.time() - start_time:.2f}s. LL: {-res_gp.fun:.4f}")
    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit the dual-process model.")
    parser.add_argument("--meta-model", choices=['evc', 'insight', 'hybrid', 'value_gap', 'belief_gap', 'mcts_only'],
                        default='evc', help="Specify the metacontroller architecture (or 'mcts_only').")
    parser.add_argument("--fitter-mode", choices=['mcts_only', 'metacontroller_fit'], required=True)
    parser.add_argument("--mcts-results-for-priors", help="Path to fitted MCTS_only results.")
    parser.add_argument("--input-file", required=True, help="Path to the full human data file (e.g., actualDim_...csv)")
    parser.add_argument("--output-dir", required=True, help="The exact, final directory to save outputs to.")
    parser.add_argument("--output-file", required=False,
                        help="Optional: Exact output CSV file path (used for unique naming in pipeline).")
    parser.add_argument("--cores", type=int, default=-1)
    parser.add_argument("--n-calls", type=int, default=100)
    parser.add_argument("--regularization-priors", help="Path to a fitted CSV for regularization.")
    parser.add_argument("--lambda-strength", type=float, default=0.0)

    args = parser.parse_args()

    group_means, group_stds = None, None
    if args.regularization_priors:
        try:
            priors_df = pd.read_csv(args.regularization_priors)
            group_means = priors_df.mean(numeric_only=True)
            group_stds = priors_df.std(numeric_only=True)
        except FileNotFoundError:
            print(f"Warning: Regularization priors file not found: '{args.regularization_priors}'.")

    mcts_priors = None
    if args.fitter_mode == 'mcts_only':
        space_to_use = MCTS_ONLY_SPACE
    elif args.fitter_mode == 'metacontroller_fit':
        if args.meta_model == 'evc':
            space_to_use = EVC_SPACE
        elif args.meta_model == 'insight':
            space_to_use = INSIGHT_SPACE
        elif args.meta_model == 'hybrid':
            space_to_use = HYBRID_SPACE
        elif args.meta_model == 'value_gap':
            space_to_use = VALUE_GAP_SPACE
        elif args.meta_model == 'belief_gap':
            space_to_use = BELIEF_GAP_SPACE
        else:
            sys.exit(f"FATAL: Unknown meta-model '{args.meta_model}'")

        if not args.mcts_results_for_priors:
            sys.exit("FATAL: MCTS priors required for metacontroller fitting.")
        try:
            mcts_priors = pd.read_csv(args.mcts_results_for_priors)
        except FileNotFoundError:
            sys.exit(f"FATAL: MCTS priors file not found: '{args.mcts_results_for_priors}'.")

    model_output_dir = args.output_dir
    os.makedirs(model_output_dir, exist_ok=True)

    if args.output_file:
        output_filename = args.output_file
    elif args.fitter_mode == 'mcts_only':
        output_filename = os.path.join(model_output_dir, "fitted_mcts_only.csv")
    else:
        output_filename = os.path.join(model_output_dir, "fitted_metacontroller.csv")

    if args.regularization_priors and args.lambda_strength > 0:
        if not output_filename.endswith('_regularized.csv'):
            output_filename = output_filename.replace('.csv', '_regularized.csv')

    p_info = prepare_data(args.input_file, load_full_df=False)
    pids = p_info['valid_prolific_ids']
    num_cores = os.cpu_count() if args.cores == -1 else args.cores

    print(f"Starting parallel fitting on {num_cores} cores for {len(pids)} participants...")
    print(f"Model: {args.meta_model}")
    print(f"Output directory: {model_output_dir}")
    start_time_total = time.time()

    results = Parallel(n_jobs=num_cores)(
        delayed(fit_participant)(
            idx, pid, args.input_file, args.n_calls, args.fitter_mode, space_to_use,
            mcts_priors,
            group_means, group_stds, args.lambda_strength, args.meta_model
        ) for idx, pid in enumerate(pids)
    )

    results_df = pd.DataFrame([r for r in results if r and 'error' not in r])
    total_time_min = (time.time() - start_time_total) / 60
    if not results_df.empty:
        results_df.to_csv(output_filename, index=False)
        print(f"\nSuccessfully processed {len(results_df)} participants.")

        base_log_name = os.path.basename(output_filename).replace('.csv', '')
        summary_log_path = os.path.join(model_output_dir, f"{base_log_name}_fitting_summary.log")
        summary_str = results_df.drop(columns=['prolific_id'], errors='ignore').describe().to_string()
        log_header = f"Fitting Summary for: {args.meta_model}\n"
        log_header += f"Data File: {args.input_file}\n"
        log_header += f"Total Participants: {len(results_df)}\n"
        log_header += f"Total Runtime: {total_time_min:.2f} minutes.\n"
        log_header += "=" * 30 + "\n"

        try:
            with open(summary_log_path, 'w') as f:
                f.write(log_header)
                f.write(summary_str)
            print(f"Fitting summary log saved to: {summary_log_path}")
        except Exception as e:
            print(f"Error: Could not write summary log. {e}")

        print("\n" + "=" * 30 + "\nSUMMARY OF FITTED PARAMETERS\n" + "-" * 30)
        print(summary_str)
        print("=" * 30 + "\n")

    print(f"\nFitting complete! Results saved to {output_filename}")
    print(f"Total runtime: {total_time_min:.2f} minutes.")