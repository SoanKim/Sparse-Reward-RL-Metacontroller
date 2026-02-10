#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim)
# Title: environment.py
# Explanation: Defines the Environment class, which encapsulates all data for a single trial.
#              It acts as a data container, providing a clean interface for agents to access
#              trial-specific information (like the relation_matrix) and check its validity.

from __future__ import annotations

from typing import List

import numpy as np


class Environment:
    def __init__(self, participant_data, trial_idx, true_set_index, attributes):
        self.attributes = attributes
        self.true_set_index = true_set_index

        self.participant_data = participant_data
        self.trial_idx = trial_idx

        self.projection_matrices = None
        self.relation_matrix = None
        self.digit_cards = None
        self._valid = True

        try:
            self.relation_matrix = participant_data['relation_matrices'][trial_idx]
            # Make projection_matrices loading optional for validity
            if 'projection_matrices' in participant_data:
                self.projection_matrices = participant_data['projection_matrices'][trial_idx]
            else:
                self.projection_matrices = None  # Okay if missing for basic function

            if 'digit_cards' in participant_data:
                self.digit_cards = participant_data['digit_cards'][trial_idx]
            else:
                self.digit_cards = None

            # Checks if relation_matrix needed for H-scores/LL is missing
            if self.relation_matrix is None or not isinstance(self.relation_matrix, np.ndarray):
                print(f"Warning: Invalid relation_matrix for trial {trial_idx}. Marking trial invalid.")
                self._valid = False

        except (KeyError, IndexError, TypeError) as e:
            print(f"Warning: Could not load data for trial {trial_idx}. Marking as invalid. Error: {e}")
            self._valid = False
            # Attributes are None if loading failed
            self.relation_matrix = None
            self.projection_matrices = None
            self.digit_cards = None

    def _construct_projection_matrices(self):
        # This logic is in preprocess.py.
        pass

    def get_triplet_relations(self, triplet_idx: int) -> List[int] | None:
        if not self.is_valid() or self.relation_matrix is None:
            return None
        try:
            return list(self.relation_matrix[triplet_idx])
        except (IndexError, TypeError):
            return None

    def get_relations_for_attribute(self, attribute: str) -> np.ndarray | None:
        if not self.is_valid() or self.relation_matrix is None:
            return None
        try:
            att_idx = self.attributes.index(attribute)
            return self.relation_matrix[:, att_idx]
        except (ValueError, TypeError, IndexError):
            return None

    def is_valid(self) -> bool:
        """
        Checks if the environment was loaded correctly and has the
        necessary data (like relation_matrix) to be used in analysis.
        """
        return self._valid
