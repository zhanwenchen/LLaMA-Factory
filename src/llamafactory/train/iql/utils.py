"""
Utility functions for IQL training.
"""

import json
from typing import List, Tuple
import torch
from torch.nn import Linear, ReLU, Sequential


# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------
Transition = Tuple[str, str, str, str, float, bool, dict]


def load_transitions_from_goldsequences(path: str) -> List[Transition]:
    """
    Load transitions from a goldSequences JSON file.

    Args:
        path: Path to the goldSequences JSON file.

    Returns:
        List of transitions in the format (state, next_state, action, next_action, reward, done, metadata).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transitions: List[Transition] = []
    for task_data in data.values():
        task_name = task_data["taskName"]
        for seq in task_data["goldActionSequences"]:
            td, var_idx, paths = seq["taskDescription"], seq["variationIdx"], seq["path"]
            if isinstance(paths, dict):
                paths = [paths]
            steps = []
            for p in paths:
                steps.append(
                    {
                        "state": p["observation"],
                        "action": p["action"],
                        "reward": float(p["score"]),
                        "done": (p["isCompleted"].lower() == "true") if isinstance(p["isCompleted"], str) else p["isCompleted"],
                    }
                )
            for i in range(len(steps) - 1):
                cur, nxt = steps[i], steps[i + 1]
                transitions.append(
                    (
                        cur["state"],  # s
                        nxt["state"],  # s'
                        cur["action"],  # a
                        nxt["action"],  # a'
                        nxt["reward"],  # r
                        nxt["done"],    # done
                        {
                            "task_name": task_name,
                            "task_description": td,
                            "variation_idx": var_idx,
                            "fold": seq["fold"],
                        },
                    )
                )
    return transitions


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int, device, dtype: torch.dtype) -> Sequential:
    """
    Create a multi-layer perceptron (MLP) network.

    Args:
        in_dim: Input dimension.
        hidden: Hidden layer dimension.
        out_dim: Output dimension.
        layers: Number of hidden layers.
        dtype: Data type for the network parameters.

    Returns:
        Sequential module representing the MLP.
    """
    seq = [Linear(in_dim, hidden, device=device, dtype=dtype), ReLU()]
    for _ in range(layers - 1):
        seq += [Linear(hidden, hidden, device=device, dtype=dtype), ReLU()]
    seq.append(Linear(hidden, out_dim, device=device, dtype=dtype))
    return Sequential(*seq)


def llama3_prompt(s: str, a: str) -> str:
    """
    Format a prompt for LLaMA-3 models.

    Args:
        s: State/observation string.
        a: Action string.

    Returns:
        Formatted prompt string.
    """
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.\n<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n" + s.strip() + "\n<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n" + a.strip() + "<|eot_id|>"
    )


def collate(batch: List[Transition]) -> Tuple:
    """
    Collate function for DataLoader to batch transitions.

    Args:
        batch: List of transitions.

    Returns:
        Tuple of (states, next_states, actions, next_actions, rewards, dones, metadata).
    """
    return list(zip(*batch))
