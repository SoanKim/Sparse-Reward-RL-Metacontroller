#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim)
# Title: mcts.py
# Explanation: Provides the core, generic implementation of the Monte Carlo Tree Search (MCTS) algorithm.


from __future__ import annotations
import random
import math
from typing import Set, FrozenSet, Tuple, Callable, Any, Optional
from node import Node
from environment import Environment


class MCTS:
    def __init__(self, handler: Any, ucb_exploration_constant: float = math.sqrt(2)):
        self.handler = handler
        self.ucb_exploration_constant = ucb_exploration_constant

    def _get_tree_stats(self, node: Node) -> Tuple[int, int]:
        if not node.children:
            return 1, 1
        depth = 0
        num_nodes = 1
        for child in node.children.values():
            child_depth, child_nodes = self._get_tree_stats(child)
            depth = max(depth, child_depth)
            num_nodes += child_nodes
        return depth + 1, num_nodes

    def _selection(self, node: Node) -> Node:
        current_node = node
        while current_node.children:
            unvisited_children = [child for child in current_node.children.values() if child.N == 0]
            if unvisited_children:
                return random.choice(unvisited_children)
            current_node = max(current_node.children.values(), key=lambda n: n.ucb1(self.ucb_exploration_constant))
        return current_node

    def _expansion(self, node: Node, current_valid_candidates: Set[int], env: Environment) -> Node:
        # Pass the environment and candidates to the handler to get truly legal actions
        available_actions = self.handler.get_available_actions(node.state, current_valid_candidates, env)

        if available_actions:
            chosen_action_to_expand = random.choice(list(available_actions))
            next_state = node.state.union({chosen_action_to_expand})
            new_child = Node(state=next_state, parent=node, action=chosen_action_to_expand)
            node.children[chosen_action_to_expand] = new_child
            return new_child
        return node

    def _backpropagate(self, node: Node, reward: float):
        temp_node = node
        while temp_node is not None:
            temp_node.N += 1
            temp_node.Q += reward
            temp_node = temp_node.parent

    def _rollout(self, node: Node, initial_candidates: Set[int], env: Environment,
                 reward_function: Callable) -> float:
        final_candidates = self.handler.rollout_policy(initial_candidates, node.state, env)
        reward = reward_function(
            initial_candidates=initial_candidates,
            final_candidates=final_candidates,
            path=node.state,
            env=env
        )
        return float(reward)

    def run_search(self, n_simulations: int, current_state: FrozenSet,
                   current_valid_candidates: Set[int], env: Environment,
                   reward_function: Callable, **kwargs) -> tuple[Optional[Any], dict]:
        mcts_root = Node(state=current_state)

        # Pass the required arguments to this initial check
        if not self.handler.get_available_actions(mcts_root.state, current_valid_candidates, env):
            return None, {}

        for _ in range(n_simulations):
            node = self._selection(mcts_root)

            # Checks if the node is expandable and has not been fully explored
            if node.N > 0 and not self.handler.is_terminal(node.state, current_valid_candidates):
                node = self._expansion(node, current_valid_candidates, env)

            rollout_reward = self.handler.rollout_policy(node.state, current_valid_candidates, env, reward_function)
            self._backpropagate(node, rollout_reward)

        if not mcts_root.children:
            return None, {}

        best_child = max(mcts_root.children.values(), key=lambda n: n.N)
        return best_child.action, {}
