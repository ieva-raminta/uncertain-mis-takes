import os
import json 
import ast
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import mannwhitneyu

def calculate_significance(preds1, preds2):
    if len(preds1) == 0 or len(preds2) == 0:
        return None
    preds1 = [ast.literal_eval(p) for p in preds1]
    preds2 = [ast.literal_eval(p) for p in preds2]
    stat, p_value = mannwhitneyu(preds1, preds2, alternative='two-sided')
    return p_value

all_results = []
def read_results(results_folder):
    results = {}
    for model in ["combined_", "linear_", "transformer_", "combined_fusion_", "combined_fusion_MLP_"]:
        lowest_value = float('inf')
        for filename in os.listdir(results_folder):
            if model+"cross" in filename:
                num = model.count('_')
                hyperparam = "_".join(filename.removesuffix(".txt").split('_')[num:])
                with open(os.path.join(results_folder, filename), 'r') as file:
                    content = file.read().strip()
                    for cont in content.split("\n"):
                        if "Gold" not in cont: 
                            type = cont.split('(')[1].split(')')[0] if '(' in cont and ')' in cont else None
                            value = cont.split(':')[-1].strip() if ':' in cont else None
                            all_results.append([value, model+hyperparam+"_"+type])
                            if float(value.strip()) < lowest_value: 
                                lowest_value = float(value.strip())
                                results[model] = [value, model+hyperparam+"_"+type]

    return all_results
    

def load_data (file_path, is_varierrnli=None):
  # read json data
  with open(file_path, "r") as f:
    data = json.load(f)

    # initialize output variabiles
    targets_soft = list()
    targets_pe = list()
    annotators_pe = list()
    annotations_possible = list() 
    ids = list()

  # loop on each item
  for id_, content in data.items():

      # extract id of the item
      ids.append(id_)

      # extract annotators of the item
      annotators_pe.append(content["annotators"])

      # extract soft label of the item and append it to the targets' soft list
      soft_label = content.get("soft_label", {})
      soft_list = list(soft_label.values())

      # extract annotators and annotations of the item and append it to the targets' PE list

      if is_varierrnli is None: # if the dataset is not varierrnli, just extract the annotations

        targets_soft.append(soft_list)

        annotation_dict = content["annotations"]
        annotations = [int(annotation_dict[ann]) for ann in content["annotators"].split(",")]
        targets_pe.append(annotations)


      else: # if the dataset is varierrnli loop on contradiction, entailment, neutral
        soft_list=[[v['0'], v['1']] for v in soft_label.values()]
        annotations = content.get("annotations", {})
        annotators = list(annotations.keys())
        num_annotators = len(annotators)

        label_to_index = ["contradiction", "entailment", "neutral"]
        # Initialize label vectors for each annotator
        label_vectors = {label: [0] * num_annotators for label in label_to_index}
        annotator_to_index = {annotator: idx for idx, annotator in enumerate(annotators)}

        # Fill in the label vectors
        for annotator, annotation_str in annotations.items():
            idx = annotator_to_index[annotator]
            for label in annotation_str.split(','):
                label = label.strip()
                if label in label_vectors:
                    label_vectors[label][idx] = 1

        targets_soft.append(soft_list)
        targets_pe.append(list(label_vectors.values()))
        poss = []
        for lab in ["contradiction", "entailment", "neutral"]:
            if lab in annotations.values():
              poss.append(1)
            else: 
              poss.append(0)
        annotations_possible.append(poss)

  return(targets_soft,targets_pe,annotators_pe,ids,data,annotations_possible)


if __name__ == "__main__":
    results_folder = 'results'
    all_results = read_results(results_folder)
    printed = []
    all_results.sort(key=lambda x: float(x[0]))
    all_results = [a for a in all_results if "With Labels" in a[1]]
    best_linear = 1
    best_transformer = 1
    best_combined = 1
    best_combined_fusion = 1
    best_combined_fusion_MLP = 1
    best_noexp = 1
    best_box = 1
    best_pe = 1
    best_linear_preds = []
    best_transformer_preds = []
    best_combined_preds = []
    best_combined_fusion_preds = []
    best_combined_fusion_MLP_preds = []
    best_noexp_preds = []
    best_box_preds = []
    best_pe_preds = []
    for value in all_results:
        if float(value[0]) > 0: #< 0.45:
            path_to_predictions = "/home/irs38/varierr/results_to_submit/"+value[1].split("With Labels")[0]+"lab_ven_test_soft.tsv"
            path_to_predictions = path_to_predictions.replace("{", "").replace("}", "")
            # read in the predictions file
            preds = []
            if os.path.exists(path_to_predictions):
                with open(path_to_predictions, 'r') as f:
                    lines = f.readlines()
                    for line in lines[1:]:
                        preds.append(line.strip().split('\t')[1])
            if os.path.exists(path_to_predictions) and path_to_predictions not in printed and float(value[0]) < 0.35:
                print("-"*30)
                print("dev score:", value[0])
                print("Path to predictions:", path_to_predictions)
                printed.append(path_to_predictions)
            if "explanations" not in value[1] and "box" not in value[1] and "none" not in value[1] and float(value[0]) < best_noexp: 
                best_noexp = float(value[0])
                best_noexp_preds = preds
            if "linear" in value[1] and float(value[0]) < best_linear:
                best_linear = float(value[0])
                best_linear_preds = preds
            if "transformer" in value[1] and float(value[0]) < best_transformer:
                best_transformer = float(value[0])
                best_transformer_preds = preds
            if "combined_fusion_MLP" in value[1] and float(value[0]) < best_combined_fusion_MLP:
                best_combined_fusion_MLP = float(value[0])
                best_combined_fusion_MLP_preds = preds
            if "combined_fusion" in value[1] and "combined_fusion_MLP" not in value[1] and float(value[0]) < best_combined_fusion:
                best_combined_fusion = float(value[0])
                best_combined_fusion_preds = preds
            if "combined_" in value[1] and "combined_fusion" not in value[1] and float(value[0]) < best_combined:
                best_combined = float(value[0])
                best_combined_preds = preds
            if "box" in value[1] and float(value[0]) < best_box:
                best_box = float(value[0])
                best_box_preds = preds
            if "none" in value[1] and "combined" not in value[1] and float(value[0]) < best_pe:
                best_pe = float(value[0])
                best_pe_preds = preds

    print("statistical significance:")
    print("Transformer vs No Explanations:", calculate_significance(best_linear_preds, best_noexp_preds))
    print("No Explanations vs no clustering:", calculate_significance(best_noexp_preds, best_pe_preds))
    print("No clustering vs Box:", calculate_significance(best_pe_preds, best_box_preds))
    print("Box vs Combined", calculate_significance(best_box_preds, best_combined_preds))
    print("Combined vs Combined Fusion", calculate_significance(best_combined_preds, best_combined_fusion_preds))
    print("Combined Fusion vs Combined Fusion MLP", calculate_significance(best_combined_fusion_preds, best_combined_fusion_MLP_preds))
    print("Combined Fusion MLP vs Linear", calculate_significance(best_combined_fusion_MLP_preds, best_linear_preds))

    # Model names and scores
    models = [
        'No SE*', 
        "No clustering (PE)*", 
        "No explanations*", 
        "No finetuning \n of Entailment model†", 
        'Combined (concat)', 
        'Combined (fusion MLP)', 
        'Combined (fusion)', 
        'No Text Encoder'
    ]
    scores = [
        best_transformer, 
        best_pe, 
        best_noexp, best_box, 
        best_combined, best_combined_fusion_MLP, best_combined_fusion, best_linear
    ]

    # Create DataFrame
    df = pd.DataFrame({
        'Model': models,
        'Manhattan Distance': scores
    })

    # Set seaborn style
    sns.set(style='whitegrid')
    sns.set_context("notebook", font_scale=1.8)

    # Plot
    plt.figure(figsize=(10, 6))  # Wider to fit long labels
    ax = sns.barplot(data=df, y='Model', x='Manhattan Distance', palette='Reds_d')
    # Set fonts and labels
    ax.set_xlabel('Manhattan Distance', fontsize=18)
    ax.set_ylabel('')
    plt.title('Best Score by Model', fontsize=20)

    # Add score labels to end of bars
    for i, (score, model) in enumerate(zip(scores, models)):
        ax.text(score + 0.01, i, f'{score:.2f}', va='center', fontsize=16)

    # Improve layout
    plt.tight_layout()
    plt.savefig('best_scores_by_model.png')
    plt.show()

    print(best_pe)

    # best_results = "_".join(all_results[0][1].split("_")[:-1])
    # genlab = "gen" if "Generated" in all_results[0][1] else "lab"
    # best_path = "results_to_submit/linear_cross_entropy_0.01_0.001_no_soft_masking_finetuned_noreg_noentpen_0.1_dropout0.5_lab_ven_dev_soft.tsv" #f"results_to_submit/{best_results}_{genlab}_ven_dev_soft.tsv"
    # with open(best_path, 'r') as file:
    #     lines = file.readlines()
    #     lists = [line.split('\t')[1].strip() for line in lines]
    # parsed_lists = [ast.literal_eval(s) for s in lists]

    # file_path = '/home/irs38/varierr/data_evaluation_phase/VariErrNLI/VariErrNLI_dev.json'
    # [targets_soft_ven_dev, targets_pe_ven_dev,annotators_pe_ven_dev,ids_ven_dev,dataVEN,annotations_possible_dev] = load_data(file_path,is_varierrnli =1)

    # file_path = '/home/irs38/varierr/data_evaluation_phase/VariErrNLI/VariErrNLI_train.json'
    # [targets_soft_ven_train, targets_pe_ven_train,annotators_pe_ven_train,ids_ven_train,dataVEN_train,annotations_possible_train] = load_data(file_path,is_varierrnli =1)

    # rounded_train = [tuple(tuple(round(x, 3) for x in pair) for pair in triplet) for triplet in targets_soft_ven_train]
    # label_counts_train = Counter(rounded_train)
    # rounded = [tuple(tuple(round(x, 3) for x in pair) for pair in triplet) for triplet in targets_soft_ven_dev]
    # label_counts = Counter(rounded)
    # all_label_counts = label_counts + label_counts_train

    # label_counts_common = {k: v for k, v in all_label_counts.items() if v >= 5}
    # label_counts_uncommon = {k: v for k, v in all_label_counts.items() if v < 5}
    # def map_to_nearest_combination(combination, label_counts_common):
    #     """
    #     Maps a combination to the nearest combination in label_counts_common.
    #     """
    #     min_distance = float('inf')
    #     nearest_combination = None
    #     for common_combination in label_counts_common.keys():
    #         distance = sum(abs(a[1] - b[1]) for a, b in zip(combination, common_combination))
    #         if distance < min_distance:
    #             min_distance = distance
    #             nearest_combination = common_combination
    #     return nearest_combination
    # labels_uncommon_mapped = [map_to_nearest_combination(comb, label_counts_common) for comb in label_counts_uncommon.keys()]
    # label_counts_uncommon_mapped = Counter(labels_uncommon_mapped)
    # label_counts_uncommon_mapped = {k: v for k, v in label_counts_uncommon_mapped.items() if v >= 5}
    # labels_reassigned = label_counts_common.copy() | label_counts_uncommon_mapped.copy()

    # for label_combo, count in sorted(labels_reassigned.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{label_combo}: {count}")
    # import pdb; pdb.set_trace()

    # for i, (parsed_list, target_soft) in enumerate(zip(parsed_lists, targets_soft_ven_dev)):
    #     if parsed_list != target_soft:
    #         id_ = ids_ven_dev[i]
    #         item = dataVEN[id_]
    #         total_diff = 0
    #         for x in range(3): 
    #             total_diff += abs(parsed_list[x][0] - target_soft[x][0])
    #         if total_diff > 1:
    #             print(f"Difference at index {i}: {parsed_list} != {target_soft}")
    #             print("total_diff:", total_diff)
    #             print(item)
    #             import pdb; pdb.set_trace()
