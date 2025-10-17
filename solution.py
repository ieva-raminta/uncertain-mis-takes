from datasets import load_dataset
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo, EntailmentGPT35, EntailmentDeberta, NonContradictionFinetuned
from torch.nn.functional import log_softmax
from collections import Counter
import numpy as np
import json
import pandas as pd
import ast
import json
import ot
import random
from itertools import product
import statistics
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import torch.nn.functional as F
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from sklearn.preprocessing import StandardScaler
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from itertools import product

parser = argparse.ArgumentParser(description="VariErrNLI Model Training")
parser.add_argument("--model", type=str, default="combined", choices=["combined", "linear", "transformer", "combined_fusion", "combined_fusion_MLP"], help="Model type to use: 'combined' for combined model, 'linear' for linear regressor")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer")
parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for the optimizer")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
parser.add_argument("--num_return", type=int, default=32, help="Number of return sequences for generation")
parser.add_argument("--loss_fn", type=str, default="cross_entropy", choices=["cross_entropy", "mse", "kldiv", "nll", "cross_label"], help="Loss function to use")
parser.add_argument("--use_soft_masking", action="store_true", help="Use soft masking for training")
parser.add_argument("--entailment_model", type=str, default="box", choices=["box", "finetuned", "finetuned_explanations", "none"], help="Entailment model to use for semantic entropy")
parser.add_argument("--temperature_annealing", action="store_true", help="Use temperature annealing for softmax in semantic entropy")
parser.add_argument("--entropy_penalty", action="store_true", help="Use entropy penalty in the loss function")
parser.add_argument("--regularise_against_mean_distribution", action="store_true", help="Regularise against the mean distribution of soft labels")
parser.add_argument("--beta", type=float, default=0.05, help="Beta parameter for entropy penalty")
parser.add_argument("--lambda_", type=float, default=0.05, help="Lambda parameter for regularisation against mean distribution")
parser.add_argument("--temperature", type=float, default=2, help="Temperature for softmax in semantic entropy")
parser.add_argument("--dropout_value", type=float, default=0.3, help="Dropout value for the model")
parser.add_argument("--scheduler", type=str, default="steplr", choices=["cosine", "linear", "steplr", "reduceonplateau"], help="Scheduler type for learning rate")
parser.add_argument("--fusion_dim", type=int, default=256, help="Hidden dimension for the model")
parser.add_argument("--linear_dim", type=int, default=16, help="Hidden dimension for the linear model")
parser.add_argument("--sum_lower_than_one_penalty", action="store_true", help="Apply penalty if sum of soft labels is lower than 1")
parser.add_argument("--combinations", action="store_true", help="Use combinations of soft labels for training")
parser.add_argument("--n_unfreeze", type=int, default=0, help="Number of layers to unfreeze in the encoder")
parser.add_argument("--class_weights", action="store_true", help="Use class weights in the loss function")
args = parser.parse_args()

labelcomb = "combinations" if args.combinations else "separate"
lowerpenalty = "lowerpen" if args.sum_lower_than_one_penalty else "nolowerpen"
regularisation = "reg" if args.regularise_against_mean_distribution else "noreg"
entropypenalty = "entpen" if args.entropy_penalty else "noentpen"
temperature = "temp" if args.temperature_annealing else "notemp"
lambda_ = "" if not args.regularise_against_mean_distribution else args.lambda_
beta = "" if not args.entropy_penalty else args.beta
num_return = args.num_return
learning_rate = args.learning_rate
weight_decay = args.weight_decay
batch_size = args.batch_size
dropout_value = float(args.dropout_value)
classweights = "" if not args.class_weights else "class_weights"

sftmsk = "soft_masking" if args.use_soft_masking else "no_soft_masking"

def extract_first_sentence(text):
    sentences = sent_tokenize(text.strip())
    return sentences[0] if sentences else text.strip()

def cross_label_entropy_loss(logits, targets, C, class_weights):
    """
    logits: [B, 3, 7]
    targets: [B, 3, 7] — one-hot or soft labels
    C: [7, 7] — similarity matrix
    class_weights: [7] — 1/frequency for each class, normalized
    """
    log_probs = F.log_softmax(logits, dim=-1)              # [B, 3, 7]
    cross_target = torch.matmul(targets, C)                # [B, 3, 7]
    cross_target = cross_target.clamp(min=1e-8)            # Avoid log(0)

    raw_loss = -(cross_target * log_probs).sum(dim=-1)     # [B, 3]

    # Get dominant class per label (argmax over original target)
    dominant_class = targets.argmax(dim=-1)                # [B, 3]
    weights = class_weights[dominant_class]                # [B, 3]

    weighted_loss = raw_loss * weights                     # [B, 3]
    return weighted_loss.mean()

def compute_class_weights_from_soft_labels(targets_soft, bin_centers=None, num_classes=7):
    """
    targets_soft: [N, 3, 2] tensor of soft labels (p_false, p_true)
    Returns: class_weights [21] (if args.combinations) or [7] (default) as inverse class frequencies
    """
    if bin_centers is None:
        bin_centers = torch.tensor([0.0, 0.25, 1/3, 0.5, 2/3, 0.75, 1.0])

    # Compute p_true for each example
    p_true = targets_soft[:, :, 1]  # shape: [N, 3]

    # Find closest bin index for each p_true value
    distances = torch.abs(p_true.unsqueeze(-1) - bin_centers.to(p_true.device))  # [N, 3, 7]
    binned = distances.argmin(dim=-1)  # [N, 3] → bin indices for each of 3 values

    if args.combinations:
        bin_centers_rounded = [round(x.item(),2) for x in bin_centers]

        all_combinations = torch.tensor([[bin_centers_rounded.index(a[1]),bin_centers_rounded.index(b[1]),bin_centers_rounded.index(c[1])] for (a,b,c) in target_classes_dict.values()])

        # Ensure same shape: [N, 1, 3] vs [21, 3] → compare each sample against all combinations
        binned_expanded = binned.unsqueeze(1)  # [N, 1, 3]
        diffs = (binned_expanded != all_combinations.unsqueeze(0)).any(dim=-1)  # [N, 21]
        class_indices = (~diffs).float().argmax(dim=1)  # find first match per sample

        counts = torch.bincount(class_indices, minlength=all_combinations.size(0)).float()
        freqs = counts / counts.sum()
        weights = 1.0 / (freqs + 1e-6)
        weights = weights / weights.sum()
        return weights  # shape: [21]
    
    else:
        # Default: handle each individual bin (7 classes)
        class_indices = binned  # [N, 3]
        counts = torch.bincount(class_indices.view(-1), minlength=num_classes).float()  # [7]
        freqs = counts / counts.sum()
        weights = 1.0 / (freqs + 1e-6)
        weights = weights / weights.sum()
        return weights  # shape: [7]



scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()

value_to_class = {
    0.00: 0,
    0.25: 1,
    0.33: 2,  # rounded 0.3333
    0.50: 3,
    0.67: 4,  # rounded 0.6666
    0.75: 5,
    1.00: 6
}
class_to_value = {v: k for k, v in value_to_class.items()}

MODEL_NAME = "facebook/bart-large-mnli"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen_model_name = "meta-llama/Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
# model = AutoModelForCausalLM.from_pretrained(gen_model_name, device_map="auto", torch_dtype="auto")

class CombinedDataset(Dataset):
    def __init__(self, main_dataset, extra_dataset):
        assert len(main_dataset) == len(extra_dataset), "Datasets must be the same length"
        self.main = main_dataset
        self.extra = extra_dataset

    def __len__(self):
        return len(self.main)

    def __getitem__(self, idx):
        main_item = self.main[idx]
        extra_item = self.extra[idx]

        # Add keys from extra_item that are not in main_item
        for k, v in extra_item.items():
            if k not in main_item:
                main_item[k] = v

        return main_item
    
class CombinedSoftLabelModel(nn.Module):
    def __init__(self, model_name, entropy_input_dim=10, hidden_dim=1024, output_dim=7):
        super().__init__()
        # Text encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size  # usually 768 or 1024

        # freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        n_unfreeze = args.n_unfreeze  # or 2, or however many layers you want to unfreeze
        for layer in self.encoder.encoder.layers[-n_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        self.entropy_processor = nn.Sequential(
            nn.Linear(entropy_input_dim, args.linear_dim),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.Linear(args.linear_dim, args.linear_dim),
        )

        if args.loss_fn == "cross_entropy" or args.loss_fn == "nll" or args.loss_fn == "cross_label":
            if args.combinations:
                output_dim = num_comb_labels
            else: 
                output_dim = 7  # 3 classes * 7 bins
        else: 
            output_dim = 2
        if args.combinations:
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_dim + args.linear_dim, args.linear_dim),
                nn.ReLU(),
                nn.Dropout(dropout_value),
                nn.Linear(args.linear_dim, output_dim)
            )
        else:
            self.classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_dim + args.linear_dim, args.linear_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_value),
                    nn.Linear(args.linear_dim, output_dim)
                )
                for _ in range(3)  # one for each NLI label
            ])

    def forward(self, input_ids, attention_mask, semantic_entropy):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        entropy_repr = self.entropy_processor(semantic_entropy)

        combined = torch.cat([cls_output, entropy_repr], dim=-1)  # [batch_size, hidden + 32]
        if args.combinations:
            logits = self.classifier(combined)
        else:
            logits = [classifier(combined) for classifier in self.classifiers]  # list of 3 tensors: [batch_size, output_dim]
            logits = torch.stack(logits, dim=1)  # [batch_size, 3, output_dim]
        return logits
    
class CombinedFusionSoftLabelModel(nn.Module):
    def __init__(self, model_name, entropy_input_dim=10, hidden_dim=1024, output_dim=7):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size

        for param in self.encoder.parameters():
            param.requires_grad = False
        n_unfreeze = args.n_unfreeze  # or 2, or however many layers you want to unfreeze
        for layer in self.encoder.encoder.layers[-n_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        self.entropy_processor = nn.Sequential(
            nn.Linear(entropy_input_dim, args.linear_dim),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.Linear(args.linear_dim, args.linear_dim),
        )

        fusion_dim = args.fusion_dim  # or any reasonable fusion size

        self.cls_proj = nn.Linear(self.hidden_dim, fusion_dim)
        self.entropy_proj = nn.Linear(args.linear_dim, fusion_dim)
        self.fusion_weights = nn.Parameter(torch.ones(2)) 

        if args.loss_fn in {"cross_entropy", "nll", "cross_label"}:
            if args.combinations: 
                output_dim = num_comb_labels
            else: 
                output_dim = 7
        else:
            output_dim = 2

        if args.combinations:
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, args.linear_dim),
                nn.ReLU(),
                nn.Dropout(dropout_value),
                nn.Linear(args.linear_dim, output_dim)
            )
        else:
            self.classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(fusion_dim, args.linear_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_value),
                    nn.Linear(args.linear_dim, output_dim)
                )
                for _ in range(3)
            ])

    def forward(self, input_ids, attention_mask, semantic_entropy):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]

        entropy_repr = self.entropy_processor(semantic_entropy)

        cls_output_proj = self.cls_proj(cls_output)            # [batch_size, fusion_dim]
        entropy_proj = self.entropy_proj(entropy_repr)         # [batch_size, fusion_dim]

        fusion_weights = torch.softmax(self.fusion_weights, dim=0)

        # Weighted fusion
        fused = fusion_weights[0] * cls_output_proj + fusion_weights[1] * entropy_proj  # [batch_size, fusion_dim]

        if args.combinations:
            logits = self.classifier(fused)
        else:

            logits = [classifier(fused) for classifier in self.classifiers]
            logits = torch.stack(logits, dim=1)  # [batch_size, 3, output_dim]
        return logits 
    
class CombinedFusionMLPSoftLabelModel(nn.Module):
    def __init__(self, model_name, entropy_input_dim=10, hidden_dim=1024, output_dim=7):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size

        for param in self.encoder.parameters():
            param.requires_grad = False
        n_unfreeze = args.n_unfreeze  # or 2, or however many layers you want to unfreeze
        for layer in self.encoder.encoder.layers[-n_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        self.entropy_processor = nn.Sequential(
            nn.Linear(entropy_input_dim, args.linear_dim),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.Linear(args.linear_dim, args.linear_dim),
        )

        self.fusion_weights = nn.Parameter(torch.ones(2))  

        if args.loss_fn in {"cross_entropy", "nll", "cross_label"}:
            if args.combinations:
                output_dim = num_comb_labels
            else:
                output_dim = 7
        else:
            output_dim = 2

        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim + args.linear_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        if args.combinations:
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, args.linear_dim),
                nn.ReLU(),
                nn.Dropout(dropout_value),
                nn.Linear(args.linear_dim, output_dim)
            )
        else:
            self.classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_dim, args.linear_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_value),
                    nn.Linear(args.linear_dim, output_dim)
                )
                for _ in range(3)
            ])

    def forward(self, input_ids, attention_mask, semantic_entropy):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]

        entropy_repr = self.entropy_processor(semantic_entropy)

        combined = torch.cat([cls_output, entropy_repr], dim=-1)
        fused = self.fusion_layer(combined)
        if args.combinations:
            logits = self.classifier(fused)
        else:
            logits = [classifier(fused) for classifier in self.classifiers]
            logits = torch.stack(logits, dim=1)  # [batch_size, 3, output_dim]
        return logits


class LinearSoftLabelRegressor(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=args.linear_dim, output_dim=7):
        super().__init__()
        if args.loss_fn == "cross_entropy" or args.loss_fn == "nll" or args.loss_fn == "cross_label":
            if args.combinations: 
                output_dim = num_comb_labels
            else:
                output_dim = 7
        elif args.loss_fn == "mse" or args.loss_fn == "kldiv":
            output_dim = 2
        if args.combinations: 
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_value),
                nn.Linear(hidden_dim, output_dim)
            )
        else: 
            self.classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_value),
                    nn.Linear(hidden_dim, output_dim)
                )
                for _ in range(3)  # one for each NLI label
            ])

    def forward(self, inp, att, x):
        if args.combinations: 
            logits = self.classifier(x)  # [batch_size, output_dim]
        else: 
            logits = [classifier(x) for classifier in self.classifiers]  # list of 3 tensors: [batch_size, output_dim]
            logits = torch.stack(logits, dim=1)  # [batch_size, 3, output_dim]
        return logits

class SoftLabelClassifier(nn.Module):
    def __init__(self, num_bins=7, num_classes=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.encoder.config.hidden_size

        for param in self.encoder.parameters():
            param.requires_grad = False
        n_unfreeze = args.n_unfreeze  # or 2, or however many layers you want to unfreeze
        for layer in self.encoder.encoder.layers[-n_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
                
        if args.loss_fn == "cross_entropy" or args.loss_fn == "nll" or args.loss_fn == "cross_label":
            if args.combinations: 
                num_bins = num_comb_labels
            else: 
                num_bins = 7
        else: 
            num_bins = 2
        if args.combinations: 
            self.classifier = nn.Linear(hidden_size, num_bins)
        else: 
            self.classifiers = nn.ModuleList([
                nn.Linear(hidden_size, num_bins) for _ in range(num_classes)
            ])

    def forward(self, input_ids, attention_mask, x):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token representation

        if args.combinations:
            logits = self.classifier(cls_output)
        else: 
            logits = [classifier(cls_output) for classifier in self.classifiers]  # list of 3 tensors (batch_size, 7)
            logits = torch.stack(logits, dim=1)  # shape: (batch_size, 3, 7)
        return logits

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


file_path = '/home/irs38/varierr/data_evaluation_phase/VariErrNLI/VariErrNLI_train.json'
[targets_soft_ven_train, targets_pe_ven_train,annotators_pe_ven_train,ids_ven_train,dataVEN_train,annotations_possible_train] = load_data(file_path,is_varierrnli =1)

file_path = '/home/irs38/varierr/data_evaluation_phase/VariErrNLI/VariErrNLI_dev.json'
[targets_soft_ven_dev, targets_pe_ven_dev,annotators_pe_ven_dev,ids_ven_dev,dataVEN,annotations_possible_dev] = load_data(file_path,is_varierrnli =1)

file_path = '/home/irs38/varierr/data_evaluation_phase/VariErrNLI/VariErrNLI_test_clear.json'
[targets_soft_ven_test, targets_pe_ven_test,annotators_pe_ven_test,ids_ven_test,dataVEN_test,annotations_possible_test] = load_data(file_path,is_varierrnli =1)


rounded_train = [tuple(tuple(round(x, 2) for x in pair) for pair in triplet) for triplet in targets_soft_ven_train]
label_counts_train = Counter(rounded_train)
rounded = [tuple(tuple(round(x, 2) for x in pair) for pair in triplet) for triplet in targets_soft_ven_dev]
label_counts = Counter(rounded)
all_label_counts = label_counts + label_counts_train
label_counts_common = {k: v for k, v in all_label_counts.items() if k[0][1]+k[1][1]+k[2][1] == 1.0}
label_counts_uncommon = {k: v for k, v in all_label_counts.items() if k[0][1]+k[1][1]+k[2][1] != 1.0}
num_comb_labels = len(label_counts_common)

def map_to_nearest_combination(combination, label_counts_common):
    """
    Maps a combination to the nearest combination in label_counts_common.
    """
    min_distance = float('inf')
    nearest_combination = None
    for common_combination in label_counts_common.keys():
        distance = sum(abs(a[1] - b[1]) for a, b in zip(combination, common_combination))
        if distance < min_distance:
            min_distance = distance
            nearest_combination = common_combination
    return nearest_combination


if args.combinations: 
    targets_soft_ven_dev_rounded = []
    for target in rounded:
        if tuple([tuple(t) for t in target]) in tuple([tuple(t) for t in label_counts_uncommon]):
            targets_soft_ven_dev_rounded.append(map_to_nearest_combination(target, label_counts_common))
        else: 
            targets_soft_ven_dev_rounded.append(target)
    targets_soft_ven_train_rounded = []
    for target in rounded_train:
        if tuple([tuple(t) for t in target]) in tuple([tuple(t) for t in label_counts_uncommon]):
            targets_soft_ven_train_rounded.append(map_to_nearest_combination(target, label_counts_common))
        else: 
            targets_soft_ven_train_rounded.append(target)
    targets_soft_ven_dev = targets_soft_ven_dev_rounded
    targets_soft_ven_train = targets_soft_ven_train_rounded

    labels_uncommon_mapped = [map_to_nearest_combination(comb, label_counts_common) for comb in label_counts_uncommon.keys()]
    label_counts_uncommon_mapped = Counter(labels_uncommon_mapped)
    labels_reassigned = label_counts_common.copy() | label_counts_uncommon_mapped.copy()
    target_classes_dict = {i: [] for i in range(num_comb_labels)}
    for i, (combination, count) in enumerate(labels_reassigned.items()):
        target_classes_dict[i] = combination
    target_classes_dict_reversed = {v: k for k, v in target_classes_dict.items()}

if args.combinations:
    label_values = torch.tensor([[a[1],b[1],c[1]] for (a,b,c) in target_classes_dict.values()])
else:
    label_values = torch.tensor([0.0, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0])
K = len(label_values)

if args.combinations: 
    class_freq = [0] * num_comb_labels
    for combination in target_classes_dict.keys():
        class_freq[combination] = labels_reassigned[target_classes_dict[combination]]
else:
    class_freq = [0] * 7
    for triplet in targets_soft_ven_train:
        for i, value in enumerate(triplet):
            class_freq[i] += 1

class_weights = 1.0 / (np.array(class_freq) + 1e-6)
class_weights = class_weights / class_weights.sum()

loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to("cuda")) if args.class_weights else torch.nn.CrossEntropyLoss() if args.loss_fn == "cross_entropy" else (
    torch.nn.MSELoss() if args.loss_fn == "mse" else (
        torch.nn.KLDivLoss(reduction='batchmean') if args.loss_fn == "kldiv" else torch.nn.NLLLoss()
    ) if args.loss_fn == "nll" else cross_label_entropy_loss if args.loss_fn == "cross_label" else None
)
random.seed(42)
torch.manual_seed(42)

def compute_C_matrix(label_values, sigma=0.25, eps=1e-8):
    """
    label_values: list or tensor of numeric values, e.g. [0, 0.25, 0.33, ...]
    sigma: controls the similarity decay (use 0.25–0.5 as a good starting point)
    """
    label_values = torch.tensor(label_values, dtype=torch.float32)
    diffs = label_values.unsqueeze(0) - label_values.unsqueeze(1)  # [num_classes, num_classes, 3]

    if args.combinations:
        sq_distances = (diffs ** 2).sum(dim=2)  # [num_classes, num_classes]
        # Gaussian similarity
        sim = torch.exp(-sq_distances / (2 * sigma ** 2))  # [num_classes, num_classes]
    else: 
        sim = torch.exp(- (diffs ** 2) / (2 * sigma ** 2))             # Gaussian kernel

    # Normalize rows to sum to 1
    row_sums = sim.sum(dim=1, keepdim=True) + eps
    C = sim / row_sums

    return C

C = compute_C_matrix(label_values)


# this function re-route to the specific soft evaluation function of the dataset

def soft_label_evaluation(dataset,targets,predictions):
  if dataset == 'MP' or dataset =='mp' or dataset =='MP':
    return(average_MD(targets,predictions))
  elif dataset == 'VEN' or dataset =='VariErrNLI' or dataset =='varierrnli':
    return(multilabel_average_MD(targets,predictions))
  elif dataset =="Par" or dataset =="par"  or dataset =="CSC" or dataset =="csc"  : # par and csc use the same soft labels evaluation functions
    return(average_WS(targets,predictions))

# this function re-route to the specific perspectivist evaluation function of the dataset

def perspectivist_evaluation(dataset,targets,predictions):
  if dataset == 'MP' or dataset =='mp' or dataset =='MP':
    return(error_rate(targets,predictions))
  elif dataset == 'VEN' or dataset =='VariErrNLI' or dataset =='varierrnli':
    return(multilabel_error_rate(targets,predictions))
  elif dataset =="Par" or dataset =='par':
    return(mean_absolute_distance(targets,predictions,11))
  elif dataset =="CSC" or dataset =='csc':
    return(mean_absolute_distance(targets,predictions,6))

def average_MD(targets, predictions):
    """
    Calculates the average Manhattan Distance (MD) between corresponding pairs of target and prediction distributions.

    Parameters:
      - targets (list of lists): A list of target distributions.
      - predictions (list of lists): A list of predicted distributions.

    Returns:
      - float: The average Manhattan Distance across all target-prediction pairs.
    """
    distances = []
    for target, prediction in zip(targets, predictions):
        # Compute the Manhattan Distance for a single pair
        distance = sum(abs(p - t) for p, t in zip(prediction, target))
        distances.append(round(distance, 5))

    # Compute and return the average Manhattan Distance
    average_distance = round(sum(distances) / len(distances), 5) if distances else 0
    return average_distance

#function for soft evaluation for dataset VariErrNLI
def multilabel_average_MD(targets,predictions):
    """
    Computes the overall soft score by averaging the average Manhattan Distances
    (MD) between predicted and target distributions for each sample.
    For each sample:
    - It uses `average_MD` to compute the average of Manhattan distances between
    corresponding target and predicted distributions within that sample.
    - It returns the mean of these average MDs across all samples.

    Parameters:
        - targets (list of list of lists): Each sample contains a list of target distributions.
        - predictions (list of list of lists): Each sample contains a list of predicted distributions.
    Returns:
        - float: The soft score, rounded to 5 decimal places.
    """
    soft_scores = [average_MD(targets[sample], predictions[sample]) for sample in range(len(targets))]
    return round(sum(soft_scores) / len(soft_scores), 5)

# function for perspectivist evaluation for dataset MP
def error_rate(targets, predictions):
    """
    Calculates the average error rate between corresponding pairs of target and
    prediction vectors. The match score is defined as 1 minus the proportion of
    correctly matched values (based on absolute error) relative to the number
    of elements in each vector.

    Parameters:
       - targets (list of lists): target vector.
       - predictions (list of lists): predicted vector.

    Returns:
       - float: The average match score across all target-prediction pairs.
    """
    match_scores = []

    for target, prediction in zip(targets, predictions):
        # Compute the total absolute error for the pair
        errors = sum(abs(t - p) for t, p in zip(target, prediction))

        # Compute a normalized match score: higher is better, 1.0 means perfect match
        match_score = round(1- ((len(target) - errors) / len(target)), 5)
        match_scores.append(match_score)

    # Return the average match score across all pairs
    return float(np.mean(match_scores))

# Function for Perspectivist evaluation for dataset VariErrNLI
def multilabel_error_rate(all_real, all_pred):
    """
    Calculates the average error rate across multiple multiclass samples.
    Each sample consists of several class vectors (lists), and for each sample,
    an average error rate is computed using the `match_score` function.

    Parameters:
      - all_real (list of list of lists): Ground-truth vectors for all samples.
      - all_pred (list of list of lists): Predicted vectors for all samples.
    Returns:
      - float: The mean of all sample-level match scores.
    """
    multiclass_er = []
    for sample in range(len(all_real)):
        multiclass_er.append(error_rate(all_real[sample], all_pred[sample]))
    return float(np.mean(multiclass_er))

# 1. Soft label evaluation for dataset VariErrNLI
targets = targets_soft_ven_dev
predictions = targets
score_soft =  multilabel_average_MD(targets,predictions)
print("Soft Label evaluation for dataset VariErrNLI: " , str(score_soft))


def renumber(ids_list):
    seen = {}
    renumbered = []
    new_num = 0
    for i, idx in enumerate(ids_list):
        if idx not in seen: 
            renumbered.append(new_num)
            seen[idx] = new_num
            new_num += 1
        else: 
            renumbered.append(seen[idx])
    return renumbered

class VariErrNLIDataset_(Dataset):
    def __init__(self, data_dict, targets_soft, annotations_possible,split=None):
        self.data = list(data_dict.values()) if type(data_dict) is dict else data_dict
        self.targets = targets_soft
        self.annotations_possible = annotations_possible
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        soft = self.targets[idx] 
        possible = torch.tensor([0, 0, 0], dtype=torch.float32)

        if "split" not in item or item["split"] == "train":
            corresponding_dataset = dataset
        elif item["split"] == "dev":
            corresponding_dataset = dev_dataset
        elif item["split"] == "test":
            corresponding_dataset = test_dataset

        corresponding_item = corresponding_dataset[idx]
        if args.entailment_model == "box":
            if len(item["other_info"]["explanations"]) < 3:
                x = 0
                while len(item["other_info"]["explanations"]) < 3:
                    item["other_info"]["explanations"].append(item["other_info"]["explanations"][x])
                    x += 1
            corresponding_item["explanations"] = [item["other_info"]["explanations"][0], item["other_info"]["explanations"][1], item["other_info"]["explanations"][2]]
        text = (
            "Premise: " + item["text"]["context"] + " "
            + "Hypothesis: " + item["text"]["statement"] + " "
            + "<explanation_contradiction>: " + corresponding_item["explanations"][0] + " "
            + "<explanation_entailment>: " + corresponding_item["explanations"][1] + " "
            + "<explanation_neutral>: " + corresponding_item["explanations"][2]
        )
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            max_length=512
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(soft, dtype=torch.float32),  # [3, 2]
            "possible": torch.tensor(possible, dtype=torch.float32)  # [3]
        }

allowed_values = [0, 0.25, 0.33, 0.5, 0.66, 0.75, 1]

def round_to_closest(value, allowed):
    return min(allowed, key=lambda x: abs(x - value))

class VariErrNLIDataset(Dataset):
    def __init__(self, data_dict, targets_soft, split):
        data_file_bkp = None
        global dataVEN_train_
        dataVEN_train_ = None
        if split == 'train':
            data_file = f"preprocessed_data/{args.entailment_model}_ven_train.pt"
            data_file_bkp = f"preprocessed_data/bkp/{args.entailment_model}_ven_train.pt"
        elif split == 'dev': 
            data_file = f"preprocessed_data/{args.entailment_model}_ven_dev.pt"
        elif split == 'test':
            data_file = f"preprocessed_data/{args.entailment_model}_ven_test.pt"
        if os.path.exists(data_file):
            preprocessed_data = torch.load(data_file)
            self.data = preprocessed_data
            self.targets = targets_soft
            if split == 'train':
                if os.path.exists(data_file_bkp):
                    preprocessed_data_bkp = torch.load(data_file_bkp)
                    self.data += preprocessed_data_bkp
                    dataVEN_train_ = self.data
                if args.entailment_model == "none" or os.path.exists(data_file_bkp):
                    targets_soft_bkp = []
                    with open('/home/irs38/ChaosNLI/data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl', 'r') as f:
                        for line in f:
                            chaositem = json.loads(line)
                            soft_label = {"contradiction": {"1": chaositem["label_dist"][2], "0": 1 - chaositem["label_dist"][2]}, 
                                        "entailment": {"1": chaositem["label_dist"][0], "0": 1 - chaositem["label_dist"][0]}, 
                                        "neutral": {"1": chaositem["label_dist"][1], "0": 1 - chaositem["label_dist"][1]}}
                            rounded_soft_label = {
                                label: {
                                    k: round_to_closest(v, allowed_values)
                                    for k, v in class_dict.items()
                                }
                                for label, class_dict in soft_label.items()
                            }
                            sft_lbl = [[rounded_soft_label["contradiction"]["0"],rounded_soft_label["contradiction"]["1"]], [rounded_soft_label["entailment"]["0"], rounded_soft_label["entailment"]["1"]], [rounded_soft_label["neutral"]["0"], rounded_soft_label["neutral"]["1"]]]
                            targets_soft_bkp.append(sft_lbl)
                    with open('/home/irs38/ChaosNLI/data/chaosNLI_v1.0/chaosNLI_snli.jsonl', 'r') as f:
                        for line in f:
                            chaositem = json.loads(line)
                            soft_label = {"contradiction": {"1": chaositem["label_dist"][2], "0": 1 - chaositem["label_dist"][2]}, 
                                        "entailment": {"1": chaositem["label_dist"][0], "0": 1 - chaositem["label_dist"][0]}, 
                                        "neutral": {"1": chaositem["label_dist"][1], "0": 1 - chaositem["label_dist"][1]}}
                            rounded_soft_label = {
                                label: {
                                    k: round_to_closest(v, allowed_values)
                                    for k, v in class_dict.items()
                                }
                                for label, class_dict in soft_label.items()
                            }  
                            sft_lbl = [[rounded_soft_label["contradiction"]["0"],rounded_soft_label["contradiction"]["1"]], [rounded_soft_label["entailment"]["0"], rounded_soft_label["entailment"]["1"]], [rounded_soft_label["neutral"]["0"], rounded_soft_label["neutral"]["1"]]]
                            targets_soft_bkp.append(sft_lbl)
                    self.targets += targets_soft_bkp
        else: 
            self.data = list(data_dict.values())
            self.targets = targets_soft

            preprocessed_data = []

            if split == 'train':
                with open('/home/irs38/ChaosNLI/data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl', 'r') as f:
                    for line in f:
                        chaositem = json.loads(line)
                        soft_label = {"contradiction": {"1": chaositem["label_dist"][2], "0": 1 - chaositem["label_dist"][2]}, 
                                      "entailment": {"1": chaositem["label_dist"][0], "0": 1 - chaositem["label_dist"][0]}, 
                                      "neutral": {"1": chaositem["label_dist"][1], "0": 1 - chaositem["label_dist"][1]}}
                        rounded_soft_label = {
                            label: {
                                k: round_to_closest(v, allowed_values)
                                for k, v in class_dict.items()
                            }
                            for label, class_dict in soft_label.items()
                        }
                        premise = chaositem["example"]["premise"]
                        hypothesis = chaositem["example"]["hypothesis"]
                        item_to_add = {"text":{"context": premise, "statement": hypothesis}, "soft_label": rounded_soft_label, "split": "train"}
                        self.data.append(item_to_add)
                with open('/home/irs38/ChaosNLI/data/chaosNLI_v1.0/chaosNLI_snli.jsonl', 'r') as f:
                    for line in f:
                        chaositem = json.loads(line)
                        soft_label = {"contradiction": {"1": chaositem["label_dist"][2], "0": 1 - chaositem["label_dist"][2]}, 
                                      "entailment": {"1": chaositem["label_dist"][0], "0": 1 - chaositem["label_dist"][0]}, 
                                      "neutral": {"1": chaositem["label_dist"][1], "0": 1 - chaositem["label_dist"][1]}}
                        rounded_soft_label = {
                            label: {
                                k: round_to_closest(v, allowed_values)
                                for k, v in class_dict.items()
                            }
                            for label, class_dict in soft_label.items()
                        }
                        premise = chaositem["example"]["premise"]
                        hypothesis = chaositem["example"]["hypothesis"]
                        item_to_add = {"text":{"context": premise, "statement": hypothesis}, "soft_label": rounded_soft_label, "split": "train"}
                        self.data.append(item_to_add)
                        
            for item in self.data: 

                example_e = (
                    "Statement: Everything can be found inside a shopping mall.\n"
                    "Context: Enter the realm of shopping malls, where everything you're looking for is available without moving your car.\n"
                    "Judgment: Entailment\n"
                    "Explanation: The context implies that the shopping mall has everything one might look for, as it can be found without moving your car.\n\n"

                    "Statement: The matter of whether or not the Mass is a sacrifice for the remission of sins is controversial.\n"
                    "Context: As for the divisive issue of whether the Mass is a sacrifice for the remission of sins, the statement affirms that Christ's death upon the cross ...\n"
                    "Judgment: Entailment\n"
                    "Explanation: The context states that the Mass being a sacrifice for the remission of sins is divisive, which can be interpreted as a synonym for controversial.\n\n"

                )
                example_n = (
                    "Statement: Most rock concerts take place in the Sultan's Pool amphitheatre.\n"
                    "Context: In the summer, the Sultan's Pool, a vast outdoor amphitheatre, stages rock concerts or other big-name events.\n"
                    "Judgment: Neutral\n"
                    "Explanation: The context does not specify whether it is most or only some rock concerts that are staged in the Sultan's Pool.\n\n"

                    "Statement: This information was developed thanks to extra federal funding.\n"
                    "Context: Additional information is provided to help managers incorporate the standards into their daily operations.\n"
                    "Judgment: Neutral\n"
                    "Explanation: The context does not indicate where the information came from, which may or may not be federal funding.\n\n"

                )
                example_c = (
                    "Statement: He had recently seen pictures depicting those things.\n"
                    "Context: He hadn't seen even pictures of such things since the few silent movies run in some of the little art theaters.\n"
                    "Judgment: Contradiction\n"
                    "Explanation: If the pronoun 'he' and the object 'those things' refer to the same things in the statement and the context, then the statement negates the context.  "

                    "Statement: Octavius Decatur Gass refers to four people. "
                    "Context: One opportunist who stayed was Octavius Decatur Gass. "
                    "Judgment: Contradiction \n"
                    "Explanation: The context names one person as Octavius Decatur Gass, and does not mention additional referrents.  "

                )
                examples = example_e + example_n + example_c

                example_e_cleaned = example_e.strip()
                statement = item["text"]["statement"].strip()
                context = item["text"]["context"].strip()
                input_e = (
                    f"You are an NLI assistant. Given a statement, context, and a judgment label (Entailment, Neutral, or Contradiction), explain why the label is appropriate.\n\n"
                    f"{example_e_cleaned.strip()}\n\n"
                    f"Now consider the following example:\n"
                    f"Statement: {statement}\n"
                    f"Context: {context}\n"
                    f"Judgment: Entailment\n"
                    f"Explanation:"
                )
                example_n_cleaned = example_n.strip()
                input_n = (
                    f"You are an NLI assistant. Given a statement, context, and a judgment label (Entailment, Neutral, or Contradiction), explain why the label is appropriate.\n\n"
                    f"{example_n_cleaned.strip()}\n\n"
                    f"Now consider the following example:\n"
                    f"Statement: {statement}\n"
                    f"Context: {context}\n"
                    f"Judgment: Neutral\n"
                    f"Explanation:"
                )
                example_c_cleaned = example_c.strip()
                input_c = (
                    f"You are an NLI assistant. Given a statement, context, and a judgment label (Entailment, Neutral, or Contradiction), explain why the label is appropriate.\n\n"
                    f"{example_c_cleaned.strip()}\n\n"
                    f"Now consider the following example:\n"
                    f"Statement: {statement}\n"
                    f"Context: {context}\n"
                    f"Judgment: Contradiction\n"
                    f"Explanation:"
                )
                examples_cleaned = examples.strip()
                input_joined = (    
                    f"You are an NLI assistant. Given a statement and context, predict the judgment label (Entailment, Neutral, or Contradiction) and explain why it is appropriate.\n\n"
                    f"{examples_cleaned.strip()}\n\n"
                    f"Now consider the following example:\n"
                    f"Statement: {statement}\n"
                    f"Context: {context}\n"
                    f"Judgment:"
                )

                #input_e = example_e + f"Statement: {item["text"]['statement']} Context: {item["text"]['context']} Judgment: Entailment Explain why this is the most likely judgment label."
                #input_n = example_n + f"Statement: {item["text"]['statement']} Context: {item["text"]['context']} Judgment: Neutral Explain why this is the most likely judgment label."
                #input_c = example_c + f"Statement: {item["text"]['statement']} Context: {item["text"]['context']} Judgment: Contradiction Explain why this is the most likely judgment label."
                #input_joined = examples + f"Statement: {item["text"]['statement']} Context: {item["text"]['context']}\nJudgment: "
                
                input_ids_e = tokenizer(input_e, return_tensors="pt").to("cuda")
                input_ids_n = tokenizer(input_n, return_tensors="pt").to("cuda")
                input_ids_c = tokenizer(input_c, return_tensors="pt").to("cuda")
                input_ids_joined = tokenizer(input_joined, return_tensors="pt").to("cuda")

                input_length_e = input_ids_e["input_ids"].shape[1]
                input_length_n = input_ids_n["input_ids"].shape[1]
                input_length_c = input_ids_c["input_ids"].shape[1]
                input_length_joined = input_ids_joined["input_ids"].shape[1]

                outputs_e = model.generate(**input_ids_e, max_new_tokens=256, num_return_sequences=num_return, do_sample=True, epsilon_cutoff=0.2,output_scores=True,return_dict_in_generate=True,pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,)
                outputs_n = model.generate(**input_ids_n, max_new_tokens=256, num_return_sequences=num_return, do_sample=True, epsilon_cutoff=0.2,output_scores=True,return_dict_in_generate=True,pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,)
                outputs_c = model.generate(**input_ids_c, max_new_tokens=256, num_return_sequences=num_return, do_sample=True, epsilon_cutoff=0.2,output_scores=True,return_dict_in_generate=True,pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,)
                outputs_joined = model.generate(**input_ids_joined, max_new_tokens=256, num_return_sequences=num_return, do_sample=True, epsilon_cutoff=0.2,output_scores=True,return_dict_in_generate=True,pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,)

                log_probs_e = []
                log_probs_n = []
                log_probs_c = []
                log_probs_joined = []
                
                sequences_e = outputs_e.sequences
                scores_e = outputs_e.scores      
                for seq_idx, sequence in enumerate(sequences_e):
                    seq_log_prob = 0.0  
                    for t, step_scores in enumerate(scores_e):
                        log_probs_t = log_softmax(step_scores, dim=-1)
                        token_id = sequence[t+input_length_e] 
                        if token_id not in tokenizer.all_special_ids:
                            seq_log_prob += log_probs_t[seq_idx, token_id].clamp(min=-1e10).item()
                    log_probs_e.append(seq_log_prob)

                sequences_n = outputs_n.sequences
                scores_n = outputs_n.scores
                for seq_idx, sequence in enumerate(sequences_n):
                    seq_log_prob = 0.0  
                    for t, step_scores in enumerate(scores_n):
                        log_probs_t = log_softmax(step_scores, dim=-1)
                        token_id = sequence[t+input_length_n] 
                        if token_id not in tokenizer.all_special_ids:
                            seq_log_prob += log_probs_t[seq_idx, token_id].clamp(min=-1e10).item()
                    log_probs_n.append(seq_log_prob)

                sequences_c = outputs_c.sequences
                scores_c = outputs_c.scores
                for seq_idx, sequence in enumerate(sequences_c):
                    seq_log_prob = 0.0  
                    for t, step_scores in enumerate(scores_c):
                        log_probs_t = log_softmax(step_scores, dim=-1)
                        token_id = sequence[t+input_length_c] 
                        if token_id not in tokenizer.all_special_ids:
                            seq_log_prob += log_probs_t[seq_idx, token_id].clamp(min=-1e10).item()
                    log_probs_c.append(seq_log_prob)

                sequences_joined = outputs_joined.sequences
                scores_joined = outputs_joined.scores
                for seq_idx, sequence in enumerate(sequences_joined):
                    seq_log_prob = 0.0  
                    for t, step_scores in enumerate(scores_joined):
                        log_probs_t = log_softmax(step_scores, dim=-1)
                        token_id = sequence[t+input_length_joined] 
                        if token_id not in tokenizer.all_special_ids:
                            seq_log_prob += log_probs_t[seq_idx, token_id].clamp(min=-1e10).item()

                    log_probs_joined.append(seq_log_prob)

                samples_e = []
                for sample in sequences_e: 
                    decoded = tokenizer.decode(sample[input_length_e:] , skip_special_tokens=True)
                    decoded = extract_first_sentence(decoded)
                    samples_e.append(decoded)
                samples_n = []
                for sample in sequences_n:
                    decoded = tokenizer.decode(sample[input_length_n:], skip_special_tokens=True)
                    decoded = extract_first_sentence(decoded)
                    samples_n.append(decoded)
                samples_c = []
                for sample in sequences_c:
                    decoded = tokenizer.decode(sample[input_length_c:], skip_special_tokens=True)
                    decoded = extract_first_sentence(decoded)
                    samples_c.append(decoded)
                samples_joined = []
                for sample in sequences_joined:
                    decoded_full = tokenizer.decode(sample[input_length_joined:], skip_special_tokens=True)
                    decoded = extract_first_sentence(decoded_full)
                    samples_joined.append(decoded)

                most_likely_sequence_e = samples_e[log_probs_e.index(max(log_probs_e))]
                most_likely_sequence_n = samples_n[log_probs_n.index(max(log_probs_n))]
                most_likely_sequence_c = samples_c[log_probs_c.index(max(log_probs_c))]

                item_counter_e = Counter(zip(samples_e, log_probs_e))
                unique_items_e, counts_e = zip(*item_counter_e.items())
                unique_samples_e, unique_logprobs_e = zip(*unique_items_e)
                item_counter_n = Counter(zip(samples_n, log_probs_n))
                unique_items_n, counts_n = zip(*item_counter_n.items())
                unique_samples_n, unique_logprobs_n = zip(*unique_items_n)
                item_counter_c = Counter(zip(samples_c, log_probs_c))
                unique_items_c, counts_c = zip(*item_counter_c.items())
                unique_samples_c, unique_logprobs_c = zip(*unique_items_c)
                item_counter_joined = Counter(zip(samples_joined, log_probs_joined))
                unique_samples_joined, unique_logprobs_joined = zip(*item_counter_joined.items())

                responses_e = unique_samples_e
                log_liks_e = unique_logprobs_e
                log_liks_agg_e = [np.mean(log_lik) for log_lik in log_liks_e]
                example_e = {}
                example_e['question'] = item["text"]['statement']
                example_e['context'] = item["text"]['context']
                responses_n = unique_samples_n
                log_liks_n = unique_logprobs_n
                log_liks_agg_n = [np.mean(log_lik) for log_lik in log_liks_n]
                example_n = {}
                example_n['question'] = item["text"]['statement']
                example_n['context'] = item["text"]['context']
                responses_c = unique_samples_c
                log_liks_c = unique_logprobs_c
                log_liks_agg_c = [np.mean(log_lik) for log_lik in log_liks_c]
                example_c = {}
                example_c['question'] = item["text"]['statement']
                example_c['context'] = item["text"]['context']
                responses_joined = unique_samples_joined
                log_liks_joined = unique_logprobs_joined
                log_liks_agg_joined = [np.mean(log_lik) for log_lik in log_liks_joined]
                example_joined = {}
                example_joined['question'] = item["text"]['statement']
                example_joined['context'] = item["text"]['context']

                num_per_label = {"Contradiction": 0,"Entailment": 0, "Neutral": 0}
                responses_joined_ordered = []
                log_liks_joined_ordered = []
                for label in ["Contradiction", "Entailment", "Neutral"]:
                    for i,response in enumerate(responses_joined):
                        if response[0].strip().startswith(label):
                            num_per_label[label] += 1
                            response_no_label = response[0].strip()[len(label):].strip().replace("Explanation:", "").strip()
                            responses_joined_ordered.append(response_no_label)
                            log_liks_joined_ordered.append(log_liks_joined[i])
                
                if args.entailment_model == "box":
                    entailment_model = EntailmentDeberta()
                else: 
                    entailment_model = NonContradictionFinetuned()

                if args.entailment_model == "none":
                    semantic_ids_all = [i for i in range(len(responses_e + responses_n + responses_c))]
                    semantic_ids_joined = [i for i in range(len(responses_joined_ordered))]
                else: 
                    semantic_ids_all = get_semantic_ids(responses_e + responses_n + responses_c, model=entailment_model,strict_entailment=False, example=item)
                    semantic_ids_joined = get_semantic_ids(responses_joined_ordered, model=entailment_model, strict_entailment=False, example=item)

                renumbered_e = renumber(semantic_ids_all[:len(responses_e)])
                renumbered_n = renumber(semantic_ids_all[len(responses_e):len(responses_e)+len(responses_n)])
                renumbered_c = renumber(semantic_ids_all[len(responses_e)+len(responses_n):])
                renumbered_en = renumber(semantic_ids_all[:len(responses_e)+len(responses_n)])
                renumbered_ec = renumber(semantic_ids_all[:len(responses_e)]+semantic_ids_all[len(responses_e)+len(responses_n):len(responses_e)+len(responses_n)+len(responses_c)])
                renumbered_nc = renumber(semantic_ids_all[len(responses_e):len(responses_e)+len(responses_n)+len(responses_c)])

                renumbered_joined_e = renumber(semantic_ids_joined[:num_per_label["Entailment"]])
                renumbered_joined_n = renumber(semantic_ids_joined[num_per_label["Entailment"]:num_per_label["Entailment"]+num_per_label["Neutral"]])
                renumbered_joined_c = renumber(semantic_ids_joined[num_per_label["Entailment"]+num_per_label["Neutral"]:])
                renumbered_joined_en = renumber(semantic_ids_joined[:num_per_label["Entailment"]+num_per_label["Neutral"]])
                renumbered_joined_ec = renumber(semantic_ids_joined[:num_per_label["Entailment"]]+semantic_ids_joined[num_per_label["Entailment"]+num_per_label["Neutral"]:num_per_label["Entailment"]+num_per_label["Neutral"]+num_per_label["Contradiction"]])
                renumbered_joined_nc = renumber(semantic_ids_joined[num_per_label["Entailment"]:num_per_label["Entailment"]+num_per_label["Neutral"]+num_per_label["Contradiction"]])

                log_likelihood_per_semantic_id_e = logsumexp_by_id(renumbered_e, log_liks_agg_e, agg='sum_normalized')
                semantic_entropy_e = predictive_entropy_rao(log_likelihood_per_semantic_id_e)
                log_likelihood_per_semantic_id_n = logsumexp_by_id(renumbered_n, log_liks_agg_n, agg='sum_normalized')
                semantic_entropy_n = predictive_entropy_rao(log_likelihood_per_semantic_id_n)
                log_likelihood_per_semantic_id_c = logsumexp_by_id(renumbered_c, log_liks_agg_c, agg='sum_normalized')
                semantic_entropy_c = predictive_entropy_rao(log_likelihood_per_semantic_id_c)
                log_likelihood_per_semantic_id_all = logsumexp_by_id(semantic_ids_all, log_liks_agg_e + log_liks_agg_n + log_liks_agg_c, agg='sum_normalized')
                semantic_entropy_all = predictive_entropy_rao(logsumexp_by_id(semantic_ids_all, log_liks_agg_e + log_liks_agg_n + log_liks_agg_c, agg='sum_normalized'))
                log_likelihood_per_semantic_id_en = logsumexp_by_id(renumbered_en, log_liks_agg_e + log_liks_agg_n, agg='sum_normalized')
                semantic_entropy_en = predictive_entropy_rao(log_likelihood_per_semantic_id_en)
                log_likelihood_per_semantic_id_ec = logsumexp_by_id(renumbered_ec, log_liks_agg_e + log_liks_agg_c, agg='sum_normalized')
                semantic_entropy_ec = predictive_entropy_rao(log_likelihood_per_semantic_id_ec)
                log_likelihood_per_semantic_id_nc = logsumexp_by_id(renumbered_nc, log_liks_agg_n + log_liks_agg_c, agg='sum_normalized')
                semantic_entropy_nc = predictive_entropy_rao(log_likelihood_per_semantic_id_nc)

                log_likelihood_per_semantic_id_joined_e = logsumexp_by_id(renumbered_joined_e, log_liks_joined_ordered[:num_per_label["Entailment"]], agg='sum_normalized')
                semantic_entropy_joined_e = predictive_entropy_rao(log_likelihood_per_semantic_id_joined_e)
                log_likelihood_per_semantic_id_joined_n = logsumexp_by_id(renumbered_joined_n, log_liks_joined_ordered[num_per_label["Entailment"]:num_per_label["Entailment"]+num_per_label["Neutral"]], agg='sum_normalized')
                semantic_entropy_joined_n = predictive_entropy_rao(log_likelihood_per_semantic_id_joined_n)
                log_likelihood_per_semantic_id_joined_c = logsumexp_by_id(renumbered_joined_c, log_liks_joined_ordered[num_per_label["Entailment"]+num_per_label["Neutral"]:], agg='sum_normalized')
                semantic_entropy_joined_c = predictive_entropy_rao(log_likelihood_per_semantic_id_joined_c)
                log_likelihood_per_semantic_id_joined_all = logsumexp_by_id(semantic_ids_joined, log_liks_joined_ordered, agg='sum_normalized')
                semantic_entropy_joined_all = predictive_entropy_rao(log_likelihood_per_semantic_id_joined_all)
                log_likelihood_per_semantic_id_joined_en = logsumexp_by_id(renumbered_joined_en, log_liks_joined_ordered[:num_per_label["Entailment"]+num_per_label["Neutral"]], agg='sum_normalized')
                semantic_entropy_joined_en = predictive_entropy_rao(log_likelihood_per_semantic_id_joined_en)
                log_likelihood_per_semantic_id_joined_ec = logsumexp_by_id(renumbered_joined_ec, log_liks_joined_ordered[:num_per_label["Entailment"]]+log_liks_joined_ordered[num_per_label["Entailment"]+num_per_label["Neutral"]:num_per_label["Entailment"]+num_per_label["Neutral"]+num_per_label["Contradiction"]], agg='sum_normalized')
                semantic_entropy_joined_ec = predictive_entropy_rao(log_likelihood_per_semantic_id_joined_ec)
                log_likelihood_per_semantic_id_joined_nc = logsumexp_by_id(renumbered_joined_nc, log_liks_joined_ordered[num_per_label["Entailment"]:num_per_label["Entailment"]+num_per_label["Neutral"]+num_per_label["Contradiction"]], agg='sum_normalized')
                semantic_entropy_joined_nc = predictive_entropy_rao(log_likelihood_per_semantic_id_joined_nc)

                semantic_entropy_joined_combs = [semantic_entropy_joined_c, semantic_entropy_joined_e, semantic_entropy_joined_n, semantic_entropy_joined_ec, semantic_entropy_joined_nc, semantic_entropy_joined_en, semantic_entropy_joined_all]
                semantic_entropy_combs = [semantic_entropy_c, semantic_entropy_e, semantic_entropy_n, semantic_entropy_ec, semantic_entropy_nc, semantic_entropy_en, semantic_entropy_all]

                prediction_e_ = len([p for p in samples_joined if p.strip().startswith("Entailment")])
                prediction_n_ = len([p for p in samples_joined if p.strip().startswith("Neutral")])
                prediction_c_ = len([p for p in samples_joined if p.strip().startswith("Contradiction")])
                prediction_e = prediction_e_ / (prediction_e_ + prediction_c_ + prediction_n_) if (prediction_e_ + prediction_c_ + prediction_n_) > 0 else 0.0
                prediction_n = prediction_n_ / (prediction_e_ + prediction_c_ + prediction_n_) if (prediction_e_ + prediction_c_ + prediction_n_) > 0 else 0.0
                prediction_c = prediction_c_ / (prediction_e_ + prediction_c_ + prediction_n_) if (prediction_e_ + prediction_c_ + prediction_n_) > 0 else 0.0

                preprocessed_data.append({
                    "predicted_labels_soft": [prediction_c, prediction_e, prediction_n], 
                    "semantic_entropy_combs": semantic_entropy_combs,
                    "semantic_entropy_joined_combs": semantic_entropy_joined_combs,
                    "explanations": [most_likely_sequence_c, most_likely_sequence_e, most_likely_sequence_n],
                    "text": item["text"]
                })
            self.data = preprocessed_data
            torch.save(self.data, data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        soft = self.targets[idx]  # [3 × 2] for contradiction/entailment/neutral
        item = self.data[idx]

        predicted_labels_soft = item['predicted_labels_soft']
        semantic_entropy_combs = item['semantic_entropy_combs']
        semantic_entropy_joined_combs = item['semantic_entropy_joined_combs']

        if "French" in item["text"]["context"]: 
            import pdb; pdb.set_trace()
        return {
            "predicted_labels_soft": torch.tensor(predicted_labels_soft, dtype=torch.float32),
            "semantic_entropy_generated": torch.tensor(semantic_entropy_combs, dtype=torch.float32), 
            "semantic_entropy_labels": torch.tensor(semantic_entropy_joined_combs, dtype=torch.float32) if semantic_entropy_joined_combs is not None else [], 
            "labels": torch.tensor(soft, dtype=torch.float32),
            "explanations": item['explanations'] if 'explanations' in item else [],
        }

def evaluate(mod, dataloader, loss_fn, semantic_entropy_key="semantic_entropy_gold"):
    mod.eval()
    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            x_ = batch["predicted_labels_soft"].to(device)  # [batch_size, 3]
            x = batch[semantic_entropy_key].to(device)          # [batch_size, input_dim]
            x = torch.cat((x_, x), dim=1)  # [batch_size, 3 + input_dim]
            inp_ids = batch["input_ids"].to(device) if "input_ids" in batch else None
            att_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
            logits = mod(inp_ids, att_mask, x)

            if args.temperature_annealing:
                temperature = float(args.temperature)
                logits = logits / temperature

            targets = batch["labels"].to(device)
            p_true = targets[:, :, 1]  # shape: (batch_size, 3)
            bin_centers = torch.tensor([0.0, 0.25, 1/3, 0.5, 2/3, 0.75, 1.0], device=targets.device)

            distances = torch.abs(p_true.unsqueeze(-1) - bin_centers)
            target_classes = distances.argmin(dim=-1) 
            pred_classes = torch.argmax(logits, dim=-1)  # shape: (batch_size, 3)

            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)

            loss = 0

            if args.loss_fn == "nll":
                for i in range(3): 
                    loss += loss_fn(log_probs[:, i, :], target_classes[:, i])
                loss /= 3
            elif args.loss_fn == "mse":
                loss = (
                    loss_fn(probs[:, 0], targets[:, 0]) +  # contradiction
                    loss_fn(probs[:, 1], targets[:, 1]) +  # entailment
                    loss_fn(probs[:, 2], targets[:, 2])    # neutral
                ) / 3
            elif args.loss_fn == "cross_entropy":
                if args.combinations:
                    target_classes = [bin_centers[i] for i in target_classes]
                    target_classes = [target_classes_dict_reversed[tuple([(round(1-round(z.item(),2),2), round(z.item(),2)) for z in i])] for i in target_classes]
                    target_classes = torch.tensor(target_classes, device=targets.device)
                    loss += loss_fn(logits, target_classes)  # logits: [batch, 3, 7] #todo redefine target_classes
                else:
                    for i in range(3):
                        loss += loss_fn(logits[:, i, :], target_classes[:, i])  # logits: [batch, 3, 7]
                loss /= 3
            elif args.loss_fn == "cross_label":
                if args.combinations: 
                    num_classes = num_comb_labels
                else:
                    num_classes = 7
                if args.combinations:
                    target_classes = [bin_centers[i] for i in target_classes]
                    target_classes = [target_classes_dict_reversed[tuple([(round(1-round(z.item(),2),2), round(z.item(),2)) for z in i])] for i in target_classes]
                    target_classes = torch.tensor(target_classes, device=targets.device)
                else:
                    target_classes = target_classes.to(torch.long)  
                targets_onehot = F.one_hot(target_classes, num_classes=num_classes).float()
                if args.combinations:
                    loss = loss_fn(logits, targets_onehot, C.to("cuda"), torch.tensor(class_weights).to("cuda"))  # logits: [batch, 3, num_comb_labels]
                else:
                    for i in range(3):
                        loss += loss_fn(logits[:, i, :].to("cuda"), targets_onehot[:, i].to("cuda"), C.to("cuda"), torch.tensor(class_weights).to("cuda"))
                loss /= 3

            elif args.loss_fn == "kldiv":
                loss = F.kl_div(log_probs, targets, reduction='batchmean')

            if args.entropy_penalty:
                entropy = -torch.sum(probs * torch.clamp(probs, min=1e-12).log(), dim=-1).mean()
                beta = float(args.beta)
                loss = loss - beta * entropy 

            if args.regularise_against_mean_distribution:
                if args.combinations:
                    pred_values = [target_classes_dict[y] for y in pred_classes.cpu().numpy()]
                    pred_values = np.array(pred_values)
                else:
                    pred_values = torch.stack([torch.stack([1 - bin_centers[pred_classes[:, i]], bin_centers[pred_classes[:, i]]], dim=1) for i in range(pred_classes.shape[1])], dim=1)
                    pred_values = pred_values.cpu().numpy()
                global_avg_per_labeltype = [torch.tensor(c_av).to("cuda"), torch.tensor(e_av).to("cuda"), torch.tensor(n_av).to("cuda")]
                global_avg_per_labeltype = torch.stack(global_avg_per_labeltype, dim=0)
                pred_values = torch.tensor(pred_values, dtype=torch.float32) if isinstance(pred_values, np.ndarray) else pred_values
                deviation = pred_values.to("cuda") - global_avg_per_labeltype.unsqueeze(0).to("cuda")  # shape: (B, 3, 2)
                penalty = -torch.norm(deviation, p=2) / pred_values.size(0)
                lambda_ = float(args.lambda_)
                loss = loss + lambda_ * penalty

            if args.sum_lower_than_one_penalty:
                pred_pos = [bin_centers[pred_classes[:, i]] for i in range(pred_classes.shape[1])]
                pred_pos_sum = torch.stack(pred_pos, dim=1).sum(dim=1)
                sum_penalty = [abs(1 - pred_pos_sum[i]) for i in range(pred_pos_sum.shape[0])]
                sum_penalty = torch.tensor(sum_penalty, device=pred_pos_sum.device).mean()
                loss = loss + sum_penalty

            total_loss += loss.item() * batch[semantic_entropy_key].size(0)
            total_samples += batch[semantic_entropy_key].size(0)

            if args.combinations:
                pred_values = [target_classes_dict[y] for y in pred_classes.cpu().numpy()]
                pred_values = np.array(pred_values)
            else:
                pred_values = torch.stack([torch.stack([1 - bin_centers[pred_classes[:, i]], bin_centers[pred_classes[:, i]]], dim=1) for i in range(pred_classes.shape[1])], dim=1)
                pred_values = pred_values.cpu().numpy()

            all_targets.extend(targets.tolist())
            all_predictions.extend(pred_values.tolist())

    avg_loss = total_loss / total_samples
    soft_score = multilabel_average_MD(all_targets, all_predictions)

    return avg_loss, soft_score, all_predictions

def evaluate_test(mod, dataloader, loss_fn, semantic_entropy_key="semantic_entropy_gold"):
    mod.eval()
    total_samples = 0

    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            x_ = batch["predicted_labels_soft"].to(device)  # [batch_size, 3]
            x = batch[semantic_entropy_key].to(device)          # [batch_size, input_dim]
            x = torch.cat((x_, x), dim=1)  # [batch_size, 3 + input_dim]
            inp_ids = batch["input_ids"].to(device) if "input_ids" in batch else None
            att_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
            logits = mod(inp_ids, att_mask, x)

            bin_centers = torch.tensor([0.0, 0.25, 1/3, 0.5, 2/3, 0.75, 1.0], device="cuda")

            pred_classes = torch.argmax(logits, dim=-1)  # shape: (batch_size, 3)

            total_samples += batch[semantic_entropy_key].size(0)

            if args.combinations:
                pred_values = [target_classes_dict[y] for y in pred_classes.cpu().numpy()]
                pred_values = np.array(pred_values)
            else:
                pred_values = torch.stack([torch.stack([1 - bin_centers[pred_classes[:, i]], bin_centers[pred_classes[:, i]]], dim=1) for i in range(pred_classes.shape[1])], dim=1)
                pred_values = pred_values.cpu().numpy()

            all_predictions.extend(pred_values.tolist())

    return all_predictions


dataset = VariErrNLIDataset(dataVEN_train, targets_soft_ven_train, split="train")
semantic_entropy_generated = scaler2.fit_transform([item["semantic_entropy_combs"] for item in dataset.data if "semantic_entropy_combs" in item])
semantic_entropy_labels = scaler3.fit_transform([item["semantic_entropy_joined_combs"] for item in dataset.data if "semantic_entropy_joined_combs" in item])
dataset.semantic_entropy_generated = torch.tensor(semantic_entropy_generated, dtype=torch.float32)
dataset.semantic_entropy_labels = torch.tensor(semantic_entropy_labels, dtype=torch.float32)
if args.entailment_model == "box":
    dataVEN_train_ = dataVEN_train.copy()
elif  args.entailment_model == "none":
    dataVEN_train_ = dataset.data.copy()
dataset_ = VariErrNLIDataset_(dataVEN_train_, targets_soft_ven_train, split="train",annotations_possible=annotations_possible_train)
combined_dataset = CombinedDataset(dataset, dataset_)
dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

dev_dataset = VariErrNLIDataset(dataVEN, targets_soft_ven_dev, split="dev")
semantic_entropy_generated = scaler2.transform([item["semantic_entropy_combs"] for item in dev_dataset.data if "semantic_entropy_combs" in item])
semantic_entropy_labels = scaler3.transform([item["semantic_entropy_joined_combs"] for item in dev_dataset.data if "semantic_entropy_joined_combs" in item])
dev_dataset.semantic_entropy_generated = torch.tensor(semantic_entropy_generated, dtype=torch.float32)
dev_dataset.semantic_entropy_labels = torch.tensor(semantic_entropy_labels, dtype=torch.float32)
if args.entailment_model == "box" or args.entailment_model == "none":
    dataVEN_dev_ = dataVEN.copy()
dev_dataset_ = VariErrNLIDataset_(dataVEN, targets_soft_ven_dev, split="dev",annotations_possible=annotations_possible_dev)
combined_dataset_dev = CombinedDataset(dev_dataset, dev_dataset_)
dev_loader = DataLoader(combined_dataset_dev, batch_size=batch_size, shuffle=False)

test_dataset = VariErrNLIDataset(dataVEN_test, targets_soft_ven_test, split="test")
semantic_entropy_generated = scaler2.transform([item["semantic_entropy_combs"] for item in test_dataset.data if "semantic_entropy_combs" in item])
semantic_entropy_labels = scaler3.transform([item["semantic_entropy_joined_combs"] for item in test_dataset.data if "semantic_entropy_joined_combs" in item])
test_dataset.semantic_entropy_generated = torch.tensor(semantic_entropy_generated, dtype=torch.float32)
test_dataset.semantic_entropy_labels = None 
if args.entailment_model == "box" or args.entailment_model == "none":
    dataVEN_test_ = dataVEN_test.copy()
test_dataset_ = VariErrNLIDataset_(dataVEN_test, targets_soft_ven_test, split="test",annotations_possible=annotations_possible_test)
combined_dataset_test = CombinedDataset(test_dataset, test_dataset_)
test_loader = DataLoader(combined_dataset_test, batch_size=batch_size, shuffle=False)

class_weights = compute_class_weights_from_soft_labels(torch.tensor(targets_soft_ven_train))

def calculate_mean_soft_label (targets):

    max_len = max(len(row) for row in targets)
    # Initialize sums and counts for each column
    sums = [0] * max_len
    counts = [0] * max_len

    # Accumulate values for each column
    for row in targets:
        for i in range(len(row)):
            sums[i] += row[i]
            counts[i] += 1

    # Compute means
    mean_soft_label = [sums[i] / counts[i] for i in range(max_len)]

    return(mean_soft_label)

c = list()
e = list()
n = list()
for k in range(len(targets_soft_ven_train)):
  c.append(targets_soft_ven_train[k][0])
  e.append(targets_soft_ven_train[k][1])
  n.append(targets_soft_ven_train[k][2])

c_av = calculate_mean_soft_label(c)
e_av = calculate_mean_soft_label(e)
n_av = calculate_mean_soft_label(n)

cen_av = [(x + y + z)/3 for x, y, z in zip(c_av, e_av, n_av)]

predictions_mostfreq_soft_ven = list()
for k in range(len(targets_soft_ven_dev)):
    predictions_mostfreq_soft_ven.append([c_av, e_av, n_av])


num_epochs = 300

# Train and eval on semantic_entropy_generated

# linear_model2 = LinearSoftLabelRegressor().to("cuda")
# transformer_model2 = SoftLabelClassifier().to("cuda")
# combined_model2 = CombinedSoftLabelModel(model_name=MODEL_NAME).to("cuda")
# combined_fusion_model2 = CombinedFusionSoftLabelModel(model_name=MODEL_NAME).to("cuda")
# combined_fusion_MLP_model2 = CombinedFusionMLPSoftLabelModel(model_name=MODEL_NAME).to("cuda")
# used_model2 = combined_model2 if args.model == "combined" else linear_model2 if args.model == "linear" else transformer_model2 if args.model == "transformer" else combined_fusion_model2 if args.model == "combined_fusion" else combined_fusion_MLP_model2
# optimizer2 = torch.optim.AdamW(used_model2.parameters(), lr=learning_rate, weight_decay=weight_decay)
# if args.scheduler == "reduce_on_plateau":
#    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'min', factor=0.5, patience=5)
# elif args.scheduler == "cosine":
#    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=num_epochs, eta_min=0.0)
# elif args.scheduler == "steplr":   
#    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=3, gamma=0.5)
# elif args.scheduler == "linear":
#    scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer2, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)

# best_val_loss2 = float('inf')
# best_soft_score2 = float('inf')
# patience = 3
# patience_counter = 0

# for epoch in range(num_epochs):
#     used_model2.train()
#     for batch in dataloader:
#         optimizer2.zero_grad()
        
#         x_ = batch["predicted_labels_soft"].to(device)  # [batch_size, 3]
#         x = batch["semantic_entropy_generated"].to(device)          # [batch_size, input_dim]
#         x = torch.cat((x_, x), dim=1)  # [batch_size, 3 + input_dim]
#         targets = batch["labels"].to(device)                   # [batch_size, 3, 2]
#         possible = batch["possible"].to(device)                # [batch_size, 3]
#         input_ids = batch["input_ids"].to(device)                # [batch_size, input_dim]
#         attention_mask = batch["attention_mask"].to(device)       # [batch_size, input_dim]
#         logits = used_model2(input_ids, attention_mask, x)
#         if args.temperature_annealing:
#             temperature = float(args.temperature)
#             logits = logits / temperature


#         targets = batch["labels"].to(device)
#         p_true = targets[:, :, 1]  # shape: (batch_size, 3)
#         bin_centers = torch.tensor([0.0, 0.25, 1/3, 0.5, 2/3, 0.75, 1.0], device=targets.device)

#         distances = torch.abs(p_true.unsqueeze(-1) - bin_centers)
#         target_classes = distances.argmin(dim=-1) 
#         pred_classes = torch.argmax(logits, dim=-1)  # shape: (batch_size, 3)

#         log_probs = F.log_softmax(logits, dim=-1)
#         probs = F.softmax(logits, dim=-1)  

#         loss2 = 0

#         if args.loss_fn == "nll":
#             for i in range(3): 
#                 loss2 += loss_fn(log_probs[:, i, :], target_classes[:, i])
#             loss2 /= 3
#         elif args.loss_fn == "mse":
#             loss2 = (
#                 loss_fn(probs[:, 0], targets[:, 0]) +  # contradiction
#                 loss_fn(probs[:, 1], targets[:, 1]) +  # entailment
#                 loss_fn(probs[:, 2], targets[:, 2])    # neutral
#             ) / 3
#         elif args.loss_fn == "cross_entropy":
#             if args.combinations:
#                 target_classes = [bin_centers[i] for i in target_classes]
#                 target_classes = [target_classes_dict_reversed[tuple([(round(1-round(z.item(),2),2), round(z.item(),2)) for z in i])] for i in target_classes]
#                 target_classes = torch.tensor(target_classes, device=targets.device)
#                 loss2 += loss_fn(logits, target_classes)  # logits: [batch, 3, 7] #todo redefine target_classes
                
#             else:
#                 for i in range(3):
#                     loss2 += loss_fn(logits[:, i, :], target_classes[:, i])  # logits: [batch, 3, 7]
#             loss2 /= 3
#         elif args.loss_fn == "cross_label":
#             if args.combinations: 
#                 num_classes = num_comb_labels
#             else:
#                 num_classes = 7
#             if args.combinations:
#                 target_classes = [bin_centers[i] for i in target_classes]
#                 target_classes = [target_classes_dict_reversed[tuple([(round(1-round(z.item(),2),2), round(z.item(),2)) for z in i])] for i in target_classes]
#                 target_classes = torch.tensor(target_classes, device=targets.device)
#             else:
#                 target_classes = target_classes.to(torch.long)  
#             targets_onehot = F.one_hot(target_classes, num_classes=num_classes).float()
#             if args.combinations:
#                 loss2 = loss_fn(logits, targets_onehot, C.to("cuda"), torch.tensor(class_weights).to("cuda"))  # logits: [batch, 3, num_comb_labels]
#             else:
#                 for i in range(3):
#                     loss2 += loss_fn(logits[:, i, :].to("cuda"), targets_onehot[:, i].to("cuda"), C.to("cuda"), torch.tensor(class_weights).to("cuda"))
#             loss2 /= 3
#         elif args.loss_fn == "kldiv":
#             loss2 = F.kl_div(log_probs, targets, reduction='batchmean')

#         if args.entropy_penalty:
#             entropy = -torch.sum(probs * torch.clamp(probs, min=1e-12).log(), dim=-1).mean()
#             beta = float(args.beta)
#             loss2 = loss2 - beta * entropy 

#         if args.regularise_against_mean_distribution:
#             if args.combinations:
#                 pred_values = [target_classes_dict[y] for y in pred_classes.cpu().numpy()]
#                 pred_values = np.array(pred_values)
#             else:
#                 pred_values = torch.stack([torch.stack([1 - bin_centers[pred_classes[:, i]], bin_centers[pred_classes[:, i]]], dim=1) for i in range(pred_classes.shape[1])], dim=1)
#                 pred_values = pred_values.cpu().numpy()
#             global_avg_per_labeltype = [torch.tensor(c_av).to("cuda"), torch.tensor(e_av).to("cuda"), torch.tensor(n_av).to("cuda")]
#             global_avg_per_labeltype = torch.stack(global_avg_per_labeltype, dim=0)
#             pred_values = torch.tensor(pred_values, dtype=torch.float32) if isinstance(pred_values, np.ndarray) else pred_values
#             deviation = pred_values.to("cuda") - global_avg_per_labeltype.unsqueeze(0).to("cuda")  # shape: (B, 3, 2)
#             penalty = -torch.norm(deviation, p=2) / pred_values.size(0)
#             lambda_ = float(args.lambda_)
#             loss2 = loss2 + lambda_ * penalty

#         if args.sum_lower_than_one_penalty:
#             pred_pos = [bin_centers[pred_classes[:, i]] for i in range(pred_classes.shape[1])]
#             pred_pos_sum = torch.stack(pred_pos, dim=1).sum(dim=1)
#             target_pos = [bin_centers[target_classes[:, i]] for i in range(target_classes.shape[1])]
#             target_pos_sum = torch.stack(target_pos, dim=1).sum(dim=1)
#             sum_penalty = [abs(target_pos_sum[i] - pred_pos_sum[i]) for i in range(pred_pos_sum.shape[0])]
#             sum_penalty = torch.tensor(sum_penalty, device=pred_pos_sum.device).mean()
#             loss2 = loss2 + sum_penalty

#         loss2.backward()
#         optimizer2.step()
    
#     val_loss2, soft_score2, _ = evaluate(used_model2, dev_loader, loss_fn, semantic_entropy_key="semantic_entropy_generated")
#     scheduler2.step(val_loss2)
    
#     print(f"Epoch {epoch}: train_loss2 = {loss2.item():.4f}, val_loss2 = {val_loss2:.4f}, soft_score2 = {soft_score2:.4f}")
    
#     if soft_score2 < best_soft_score2:
#         best_soft_score2 = soft_score2
#         patience_counter = 0
#         torch.save(used_model2.state_dict(), "models/best_model2.pt")
#     else:
#         patience_counter += 1
#         if patience_counter >= patience:
#             print("Early stopping triggered.")
#             break


# used_model2.load_state_dict(torch.load("models/best_model2.pt"))

# val_loss2, soft_score2, val_predictions2 = evaluate(used_model2, dev_loader, loss_fn, semantic_entropy_key="semantic_entropy_generated")
# print(f"Validation Loss (Generated): {val_loss2:.4f}, Soft Score2 (Avg MD): {soft_score2:.5f}")
# test_predictions2 = evaluate_test(used_model2, test_loader, loss_fn, semantic_entropy_key="semantic_entropy_generated")

# with open(f"results/{args.model}_{args.loss_fn}_{args.learning_rate}_{args.weight_decay}_{sftmsk}_{args.entailment_model}_{regularisation}{lambda_}_{entropypenalty}{beta}_{temperature}_dropout{dropout_value}_scheduler{args.scheduler}_lineardim{args.linear_dim}_fusiondim{args.fusion_dim}_{lowerpenalty}_{labelcomb}_unfreeze{args.n_unfreeze}_{classweights}.txt", "a") as f:
#     f.write(f"Validation Loss (Generated): {val_loss2:.4f}, Soft Score2 (Avg MD): {soft_score2:.5f}\n")

# Train and eval on semantic_entropy_generated_with_given_labels

linear_model3 = LinearSoftLabelRegressor().to("cuda")
transformer_model3 = SoftLabelClassifier().to("cuda")
combined_model3 = CombinedSoftLabelModel(model_name=MODEL_NAME).to("cuda")
combined_fusion_model3 = CombinedFusionSoftLabelModel(model_name=MODEL_NAME).to("cuda")
combined_fusion_MLP_model3 = CombinedFusionMLPSoftLabelModel(model_name=MODEL_NAME).to("cuda")
used_model3 = combined_model3 if args.model == "combined" else linear_model3 if args.model == "linear" else transformer_model3 if args.model == "transformer" else combined_fusion_model3 if args.model == "combined_fusion" else combined_fusion_MLP_model3
optimizer3 = torch.optim.AdamW(used_model3.parameters(), lr=learning_rate, weight_decay=weight_decay)
if args.scheduler == "reduce_on_plateau":
   scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer3, 'min', factor=0.5, patience=5)
elif args.scheduler == "cosine":
   scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=num_epochs, eta_min=0.0)
elif args.scheduler == "steplr":   
   scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=3, gamma=0.5)
elif args.scheduler == "linear":
   scheduler3 = torch.optim.lr_scheduler.LinearLR(optimizer3, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)

best_val_loss3 = float('inf')
best_soft_score3 = float('inf')
patience = 3
patience_counter = 0

for epoch in range(num_epochs):
    used_model3.train()
    for batch in dataloader:
        optimizer3.zero_grad()
        
        x = batch["semantic_entropy_labels"].to(device)          # [batch_size, input_dim]
        x_ = batch["predicted_labels_soft"].to(device)
        x = torch.cat((x_, x), dim=1)  # [batch_size, 3 + input_dim]
        targets = batch["labels"].to(device)                   # [batch_size, 3, 2]
        possible = batch["possible"].to(device) if "possible" in batch else None  # [batch_size, 3]
        input_ids = batch["input_ids"].to(device) if "input_ids" in batch else None  # [batch_size, input_dim]
        attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None  # [batch_size, input_dim]

        logits = used_model3(input_ids, attention_mask, x)
        if args.temperature_annealing:
            temperature = float(args.temperature)
            logits = logits / temperature

        targets = batch["labels"].to(device)
        p_true = targets[:, :, 1]  # shape: (batch_size, 3)
        bin_centers = torch.tensor([0.0, 0.25, 1/3, 0.5, 2/3, 0.75, 1.0], device=targets.device)

        distances = torch.abs(p_true.unsqueeze(-1) - bin_centers)
        target_classes = distances.argmin(dim=-1) 
        pred_classes = torch.argmax(logits, dim=-1)  # shape: (batch_size, 3)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)  

        loss3 = 0

        if args.loss_fn == "nll":
            for i in range(3): 
                loss3 += loss_fn(log_probs[:, i, :], target_classes[:, i])
            loss3 /= 3
        elif args.loss_fn == "mse":
            loss3 = (
                loss_fn(probs[:, 0], targets[:, 0]) +  # contradiction
                loss_fn(probs[:, 1], targets[:, 1]) +  # entailment
                loss_fn(probs[:, 2], targets[:, 2])    # neutral
            ) / 3
        elif args.loss_fn == "cross_entropy":
            if args.combinations:
                target_classes = [bin_centers[i] for i in target_classes]
                target_classes = [target_classes_dict_reversed[tuple([(round(1-round(z.item(),2),2), round(z.item(),2)) for z in i])] for i in target_classes]
                target_classes = torch.tensor(target_classes, device=targets.device)
                loss3 += loss_fn(logits, target_classes)  # logits: [batch, 3, 7] #todo redefine target_classes
            else:
                for i in range(3):
                    loss3 += loss_fn(logits[:, i, :], target_classes[:, i])  # logits: [batch, 3, 7]
            loss3 /= 3
        elif args.loss_fn == "cross_label":
            if args.combinations: 
                num_classes = num_comb_labels
            else:
                num_classes = 7
            if args.combinations:
                target_classes = [i if i not in label_counts_uncommon else map_to_nearest_combination(i, label_counts_common) for i in target_classes]
                target_classes = [target_classes_dict_reversed[tuple([(round(1-round(z.item(),2),2), round(z.item(),2)) for z in i])] for i in target_classes]
                target_classes = torch.tensor(target_classes, device=targets.device)
            else:
                target_classes = target_classes.to(torch.long)  
            targets_onehot = F.one_hot(target_classes, num_classes=num_classes).float()
            if args.combinations:
                loss3 = loss_fn(logits, targets_onehot, C.to("cuda"), torch.tensor(class_weights).to("cuda"))  # logits: [batch, 3, num_comb_labels]
            else:
                for i in range(3):
                    loss3 += loss_fn(logits[:, i, :].to("cuda"), targets_onehot[:, i].to("cuda"), C.to("cuda"), torch.tensor(class_weights).to("cuda"))
            loss3 /= 3
        elif args.loss_fn == "kldiv":
            loss3 = F.kl_div(log_probs, targets, reduction='batchmean')

        if args.entropy_penalty:
            entropy = -torch.sum(probs * torch.clamp(probs, min=1e-12).log(), dim=-1).mean()
            beta = float(args.beta)
            loss3 = loss3 - beta * entropy 

        if args.regularise_against_mean_distribution:
            if args.combinations:
                pred_values = [target_classes_dict[y] for y in pred_classes.cpu().numpy()]
                pred_values = np.array(pred_values)
            else:
                pred_values = torch.stack([torch.stack([1 - bin_centers[pred_classes[:, i]], bin_centers[pred_classes[:, i]]], dim=1) for i in range(pred_classes.shape[1])], dim=1)
                pred_values = pred_values.cpu().numpy()
            global_avg_per_labeltype = [torch.tensor(c_av).to("cuda"), torch.tensor(e_av).to("cuda"), torch.tensor(n_av).to("cuda")]
            global_avg_per_labeltype = torch.stack(global_avg_per_labeltype, dim=0)
            pred_values = torch.tensor(pred_values, dtype=torch.float32) if isinstance(pred_values, np.ndarray) else pred_values
            deviation = pred_values.to("cuda") - global_avg_per_labeltype.unsqueeze(0).to("cuda")  # shape: (B, 3, 2)
            penalty = -torch.norm(deviation, p=2) / pred_values.size(0)
            lambda_ = float(args.lambda_)
            loss3 = loss3 + lambda_ * penalty

        if args.sum_lower_than_one_penalty:
            pred_pos = [bin_centers[pred_classes[:, i]] for i in range(pred_classes.shape[1])]
            pred_pos_sum = torch.stack(pred_pos, dim=1).sum(dim=1)
            target_pos = [bin_centers[target_classes[:, i]] for i in range(target_classes.shape[1])]
            target_pos_sum = torch.stack(target_pos, dim=1).sum(dim=1)
            sum_penalty = [abs(target_pos_sum[i] - pred_pos_sum[i]) for i in range(pred_pos_sum.shape[0])]
            sum_penalty = torch.tensor(sum_penalty, device=pred_pos_sum.device).mean()
            loss3 = loss3 + sum_penalty

        loss3.backward()
        optimizer3.step()
    
    val_loss3, soft_score3, _ = evaluate(used_model3, dev_loader, loss_fn, semantic_entropy_key="semantic_entropy_labels")
    scheduler3.step(val_loss3)
    
    print(f"Epoch {epoch}: train_loss3 = {loss3.item():.4f}, val_loss3 = {val_loss3:.4f}, soft_score3 = {soft_score3:.4f}")
    
    if soft_score3 < best_soft_score3:
        best_soft_score3 = soft_score3
        patience_counter = 0
        torch.save(used_model3.state_dict(), "models/best_model3.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

used_model3.load_state_dict(torch.load("models/best_model3.pt"))

val_loss3, soft_score3, val_predictions3 = evaluate(used_model3, dev_loader, loss_fn, semantic_entropy_key="semantic_entropy_labels")
print(f"Validation Loss (With Labels): {val_loss3:.4f}, Soft Score (Avg MD): {soft_score3:.5f}")
test_predictions3 = evaluate_test(used_model3, test_loader, loss_fn, semantic_entropy_key="semantic_entropy_labels")

with open(f"results/{args.model}_{args.loss_fn}_{args.learning_rate}_{args.weight_decay}_{sftmsk}_{args.entailment_model}_{regularisation}{lambda_}_{entropypenalty}{beta}_{temperature}_dropout{dropout_value}_scheduler{args.scheduler}_lineardim{args.linear_dim}_fusiondim{args.fusion_dim}_{lowerpenalty}_{labelcomb}_unfreeze{args.n_unfreeze}_{classweights}.txt", "a") as f:
    f.write(f"Validation Loss (With Labels): {val_loss3:.4f}, Soft Score3 (Avg MD): {soft_score3:.5f}\n")

# df = pd.DataFrame()
# df['id'] = ids_ven_dev
# df['soft_pred'] = val_predictions2
# df.to_csv(f'results_to_submit/{args.model}_{args.loss_fn}_{args.learning_rate}_{args.weight_decay}_{sftmsk}_{args.entailment_model}_{regularisation}{lambda_}_{entropypenalty}{beta}_{temperature}_dropout{dropout_value}_scheduler{args.scheduler}_lineardim{args.linear_dim}_fusiondim{args.fusion_dim}_{lowerpenalty}_{labelcomb}_unfreeze{args.n_unfreeze}_{classweights}_gen_ven_dev_soft.tsv', sep ='\t',header= False, index= False)

# df = pd.DataFrame()
# df['id'] = ids_ven_test
# df['soft_pred'] = test_predictions2
# df.to_csv(f'results_to_submit/{args.model}_{args.loss_fn}_{args.learning_rate}_{args.weight_decay}_{sftmsk}_{args.entailment_model}_{regularisation}{lambda_}_{entropypenalty}{beta}_{temperature}_dropout{dropout_value}_scheduler{args.scheduler}_lineardim{args.linear_dim}_fusiondim{args.fusion_dim}_{lowerpenalty}_{labelcomb}_unfreeze{args.n_unfreeze}_{classweights}_gen_ven_test_soft.tsv', sep ='\t',header= False, index= False)

df = pd.DataFrame()
df['id'] = ids_ven_dev
df['soft_pred'] = val_predictions3
df.to_csv(f'results_to_submit/{args.model}_{args.loss_fn}_{args.learning_rate}_{args.weight_decay}_{sftmsk}_{args.entailment_model}_{regularisation}{lambda_}_{entropypenalty}{beta}_{temperature}_dropout{dropout_value}_scheduler{args.scheduler}_lineardim{args.linear_dim}_fusiondim{args.fusion_dim}_{lowerpenalty}_{labelcomb}_unfreeze{args.n_unfreeze}_{classweights}_lab_ven_dev_soft.tsv', sep ='\t',header= False, index= False)   

df = pd.DataFrame()
df['id'] = ids_ven_test
df['soft_pred'] = test_predictions3
df.to_csv(f'results_to_submit/{args.model}_{args.loss_fn}_{args.learning_rate}_{args.weight_decay}_{sftmsk}_{args.entailment_model}_{regularisation}{lambda_}_{entropypenalty}{beta}_{temperature}_dropout{dropout_value}_scheduler{args.scheduler}_lineardim{args.linear_dim}_fusiondim{args.fusion_dim}_{lowerpenalty}_{labelcomb}_unfreeze{args.n_unfreeze}_{classweights}_lab_ven_test_soft.tsv', sep ='\t',header= False, index= False)
df = pd.DataFrame()
