import os
import json
from collections import Counter
all_results = []

import ast

def read_soft_labels_from_tsv(filepath):
    soft_labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue  # skip malformed lines
            try:
                label_list = ast.literal_eval(parts[1])
                soft_labels.append(label_list)
            except (SyntaxError, ValueError):
                print(f"Warning: could not parse line: {line}")
    return soft_labels

original_data = []
with open("ven_test_soft.tsv", "r", encoding="utf-8") as f:
    for line in f:
        idx, trio_str = line.strip().split('\t')
        trio = eval(trio_str)  # or use `ast.literal_eval` for safety
        original_data.append((int(idx), trio))

filepath = "ven_test_soft.tsv"
labels = read_soft_labels_from_tsv(filepath)

import itertools
import numpy as np

def closest_valid_label_trio(trio, allowed_vals=[0, 0.25, 0.33, 0.5, 0.75, 1], tol=1e-3):
    original = np.array([x[1] for x in trio])
    if abs(original.sum() - 1.0) < tol:
        return trio  # already valid

    # Generate all valid combinations of second values whose sum is close to 1
    candidates = []
    for combo in itertools.product(allowed_vals, repeat=3):
        if abs(sum(combo) - 1.0) < tol:
            candidates.append(combo)

    # Find the candidate closest to the original
    best_combo = min(candidates, key=lambda c: np.linalg.norm(original - np.array(c)))
    
    # Convert back to [1 - x, x] form
    adjusted_trio = [[1 - v, v] for v in best_combo]
    return adjusted_trio

adjusted_trios = []
for trio in labels: 
    adjusted_trio = closest_valid_label_trio(trio)
    adjusted_trios.append(adjusted_trio)

output_filepath = "/home/irs38/varierr/results_to_submit/combined_fusion_MLP_cross_label_1e-05_1.0000000000000002e-06_no_soft_masking_finetuned_explanations_noreg0.05_entpen0.05_notemp_dropout0.1_schedulersteplr_lineardim16_fusiondim256_nolowerpen_separate_unfreeze3__lab_ven_test_soft.tsv"
with open(output_filepath, "w", encoding="utf-8") as f:
    for (idx, _), adjusted_trio in zip(original_data, adjusted_trios):
        f.write(f"{idx}\t{adjusted_trio}\n")