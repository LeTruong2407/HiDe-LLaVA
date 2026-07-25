#!/usr/bin/env python3
"""Summarize pairwise principal-angle overlap from a consensus checkpoint."""

import argparse
import json
import os
import re
from collections import defaultdict

import torch


LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def mean_principal_cosine(left, right):
    rank = min(left.shape[1], right.shape[1])
    if rank == 0:
        return None
    singular_values = torch.linalg.svdvals(
        left[:, :rank].float().T @ right[:, :rank].float()
    )
    return float(singular_values.mean().item())


def analyze(state):
    module_results = {}
    aggregates = defaultdict(list)

    for name, module_state in state.items():
        bases = module_state.get("task_bases", [])
        pairwise = {}
        for left_index in range(len(bases)):
            for right_index in range(left_index + 1, len(bases)):
                score = mean_principal_cosine(
                    bases[left_index], bases[right_index]
                )
                pairwise[f"{left_index}-{right_index}"] = score

        if not pairwise:
            continue

        module_mean = sum(pairwise.values()) / len(pairwise)
        match = LAYER_PATTERN.search(name)
        layer = match.group(1) if match else "unknown"
        projection = name.rsplit(".", 1)[-1]
        aggregates[f"layer_{layer}"].append(module_mean)
        aggregates[f"projection_{projection}"].append(module_mean)
        module_results[name] = {
            "task_count": len(bases),
            "pairwise_overlap": pairwise,
            "mean_overlap": module_mean,
        }

    aggregate_results = {
        key: {
            "module_count": len(values),
            "mean_overlap": sum(values) / len(values),
        }
        for key, values in sorted(aggregates.items())
        if values
    }
    return {"aggregates": aggregate_results, "modules": module_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        help="Task output directory or path to consensus_subspaces.pt",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    state_path = args.checkpoint
    if os.path.isdir(state_path):
        state_path = os.path.join(state_path, "consensus_subspaces.pt")
    state = torch.load(state_path, map_location="cpu")
    results = analyze(state)
    rendered = json.dumps(results, indent=2, sort_keys=True)

    if args.output:
        with open(args.output, "w") as handle:
            handle.write(rendered)
            handle.write("\n")
    print(rendered)


if __name__ == "__main__":
    main()
