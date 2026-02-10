#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim)
# Title: node.py
# Explanation: Defines the Node class, the fundamental data structure for the Monte Carlo Tree Search (MCTS).
#              Each node represents a state in the search process, defined by the set of attributes already analyzed.

from __future__ import annotations

import math
from typing import Dict, Optional, FrozenSet


class Node:
    def __init__(self, state: FrozenSet[int], parent: Optional[Node] = None, action: Optional[int] = None):
        self.state: FrozenSet[int] = state
        self.parent: Optional[Node] = parent
        self.action: Optional[int] = action
        self.children: Dict[int, Node] = {}
        self.N: int = 0
        self.Q: float = 0.0

    # The average reward (Q/N) for this node
    def get_value(self) -> float:
        if self.N == 0:
            return 0.0
        return self.Q / self.N

    def ucb1(self, C: float = math.sqrt(2)) -> float:
        if self.N == 0:
            return float('inf')  # Exploring unvisited nodes first
        if self.parent is None or self.parent.N == 0:
            return self.get_value()
        return self.get_value() + C * math.sqrt(math.log(self.parent.N) / self.N)

    # A string representation for debugging
    def __repr__(self) -> str:
        state_str = ", ".join(sorted([str(s) for s in self.state])) if self.state else "Root"
        return (f"Node(state={{{state_str}}}, "
                f"value={self.get_value():.2f}, Q={self.Q:.2f}, N={self.N})")