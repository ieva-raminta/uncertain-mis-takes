from datasets import load_dataset
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo, EntailmentGPT35, EntailmentDeberta
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
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from huggingface_hub import login

"""
INPUT VARIABLES:
- file_path: file path to the JSON data in your Google Drive
- is_varierrnli: optional, set it to 1 if you are evaluation the VariErrNLI dataset. Otherwise don't pass it to the function (default is None).


OUTPUT VARIABLES:

- targets_soft:
  Type: list of lists (Csc, MP, Par), list of lists of lists (VariErrNLI)
  Content: a list containg a list for each item flat integer labels representing the true soft labels distribution (i.e., the "targets"), a value for each soft label. In the case of VariErrNLI is further nested into 3 other lists (for Contradiction, Entailment, Neutral)
  Details:
    targets_soft_mp  = [[0.4,0.6], [...] ,[0.25,0.75]]
          ie: [[item1_SoftLabel"0", item1_SoftLabel"1"], ... ,[item-n_SoftLabel"0",item-n_SoftLabel"1"]]

    targets_soft_par = [[0.0, 0.25, 0.25, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0], ... ,[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]]
          ie: [[item1_SL"-5", item1_SL"-4", item1_SL"-3", item1_SL"-2", item1_SL"-1", item1_SL"0", item1_SL"1", item1_SL"2", item1_SL"3", item1_SL"4", item1_SL"5"],[...] ,[item-n_SL"-5", item-n_SL"-4", item-n_SL"-3", item-n_SL"-2", item-n_SL"-1", item-n_SL"0", item-n_SL"1", item-n_SL"2", item-n_SL"3", item-n_SL"4", item-n_SL"5"]]

    targets_soft_csc = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], ... ,[0.0, 0.0, 0.5, 0.16666666666666666, 0.3333333333333333, 0.0, 0.0]]
          ie: [[item1_SL"0", item1_SL"1", item1_SL"2", item1_SL"3", item1_SL"4", item1_SL"5", item1_SL"6"], ... ,[item-n_SL"0", item-n_SL"1", item-n_SL"2", item-n_SL"3", item-n_SL"4", item-n_SL"5", item-n_SL"6"]]

    targets_soft_ven = [[[1.0, 0.0], [0.5, 0.5], [0.25, 0.75]], ... ,[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]
          ie: [[[item1_Contradiction_SoftLabel"0", item1_Contradiction_SoftLabel"1"],[item1_Entailment_SoftLabel"0", item1_Entailment_SoftLabel"1"],[item1_Neutral_SoftLabel"0", item1_Neutral_SoftLabel"1"]], ... ,[[item-n_Contradiction_SoftLabel"0", item-n_Contradiction_SoftLabel"1"],[item-n_Entailment_SoftLabel"0", item-n_Entailment_SoftLabel"1"],[item-n_Neutral_SoftLabel"0", item-n_Neutral_SoftLabel"1"]]]

- targets_pe:
  Type: list of lists (Csc, MP, Par), list of lists of lists (VariErrNLI)
  Content: Each item is a flat list of integer labels (e.g., [0, 1, 2]) representing the annotation from each annotator (specified in annotators_pe) for the specific item.
  Details:
    targets_pe_csc = [[1,1,1,1],...,[2,4,2,2,3,4]]
    targets_pe_mp  = [[0,1,1,1,0],...,[1,1,1,0]]
    targets_pe_par = [[-4,-3,0,-1],...,[5,4,4,5]]
    targets_pe_ven = [[[0,0,0,0],[0,1,0,1], [1,1,1,0]],...,[[0,0,0,0],[1,1,1,1],[0,0,0,0]]]
  Note:
    csc, mp, par: [[Annotation for item1 by first annotator of item1, ..., Annotation for item1 by n-annotator of the item 1 ],...,[Annotation for item-n by first annotator of item-n, ... , Annotation for item-n by n-annotator of item-n ]]
    vari_err_nli: [[[Annotation for contradiction for item1 by first annotator of item1, ..., Annotation for contradiction for item1 by n-annotator of the item 1 ],[Annotation for entilment for item1 by first annotator of item1, ..., Annotation for entilment for item1 by n-annotator of the item 1 ],[Annotation for neutral for item1 by first annotator of item1, ..., Annotation for neutral for item1 by n-annotator of the item 1 ]],...,[[Annotation for contradiction for item-n by first annotator of item-n, ..., Annotation for contradiction for item-n by n-annotator of the item 1 ],[Annotation for entilment for item-n by first annotator of item-n, ..., Annotation for entilment for item-n by n-annotator of the item 1 ],[Annotation for neutral for item-n by first annotator of item-n, ..., Annotation for neutral for item-n by n-annotator of the item 1 ]]]

- annotators_pe:
  Type: list of strings
  Contents: List of annotator ID strings for each data item.
  Details: They are extracted directly from the "annotators" field in the JSON data.
    annotators_pe_csc = ['Ann844,Ann845,Ann846,Ann847', ...,'Ann60,Ann61,Ann62,Ann63,Ann64,Ann65']
    annotators_pe_mp  = ['Ann0,Ann20,Ann59,Ann62,Ann63',...,'Ann499,Ann500,Ann501,Ann505']
    annotators_pe_par = ['Ann1,Ann2,Ann3,Ann4',...,'Ann1,Ann2,Ann3,Ann4']
    annotators_pe_ven = ['Ann1,Ann2,Ann3,Ann4',...,'Ann1,Ann2,Ann3,Ann4']

- ids:
  Type: list
  Contents: Unique IDs of each example in the dataset.
  Details:These are the keys from the JSON data (e.g., "123", "456").

- data
  Type: dict
  Contents: The full parsed JSON content from the input file.

"""

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


file_path = '/home/irs38/varierr/data_practice_phase/VariErrNLI/VariErrNLI_train.json'
[targets_soft_ven_train, targets_pe_ven_train,annotators_pe_ven_train,ids_ven_train,dataVEN_train,annotations_possible_train] = load_data(file_path,is_varierrnli =1)

file_path = '/home/irs38/varierr/data_practice_phase/VariErrNLI/VariErrNLI_dev.json'
[targets_soft_ven_dev, targets_pe_ven_dev,annotators_pe_ven_dev,ids_ven_dev,dataVEN,annotations_possible_dev] = load_data(file_path,is_varierrnli =1)


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

# alternatively you can use the function
# score_soft = soft_label_evaluation('VEN',targets,predictions)


# 2. Perspectivist evaluation for dataset MP
targets = targets_pe_ven_dev
predictions = targets
score_pe = multilabel_error_rate(targets,predictions)
print("Perspectivist evaluation for dataset VariErrNLI: ", str(score_pe))

# alternatively you can use the function
# score_pe = perspectivist_evaluation('VEN',targets,predictions)

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

#VariErrNLI

c = list()
e = list()
n = list()
for k in range(len(targets_soft_ven_dev)):
  c.append(targets_soft_ven_train[k][0])
  e.append(targets_soft_ven_train[k][1])
  n.append(targets_soft_ven_train[k][2])
c_av = calculate_mean_soft_label(c)
e_av = calculate_mean_soft_label(e)
n_av = calculate_mean_soft_label(n)
print('Average soft label distribution for VariErrNLI - CONTRADICTION: ')
print( str(c_av ))
print('Average soft label distribution for VariErrNLI - ENTAILMENT:')
print(str(e_av ))
print('Average soft label distribution for VariErrNLI - NEUTRAL:')
print(str(n_av ))

predictions_mostfreq_soft_ven = list()
for k in range(len(targets_soft_ven_dev)):
  predictions_mostfreq_soft_ven.append([c_av, e_av, n_av])

print('Soft label evaluation score: ' + str(soft_label_evaluation('VEN',targets_soft_ven_dev, predictions_mostfreq_soft_ven  )  ))

"""
Pipeline for most frequent baseline for perspectivist evaluation calculation
- Step 1: calculate the most frequent label assigned by each annotator.
- Step 2: to each annotation of each annotator, assign its relative most frequently assigned label
- Step 3: calculate perspectivist score
"""


def most_frequent_pe(annotators_,targets_,annotators_topredict):

  # Step 1: Build label history per annotator
  annotator_label_history = {}

  for annotator_str, label_list in zip(annotators_, targets_):
      annotators = annotator_str.split(',')
      for ann, lbl in zip(annotators, label_list):
          if ann not in annotator_label_history:
              annotator_label_history[ann] = []
          annotator_label_history[ann].append(lbl)

  # Step 2: Function to get the most frequent label
  def most_frequent(lst):
      freq = {}
      for item in lst:
          freq[item] = freq.get(item, 0) + 1
      return max(freq.items(), key=lambda x: x[1])[0]

  # Step 3: Map each annotator to their most frequent label
  annotator_to_most_common_label = {
      ann: most_frequent(labels)
      for ann, labels in annotator_label_history.items()
  }

  # Step 4: Build new targets-like structure
  targets_most_freq = []

  for annotator_str in annotators_topredict:
      annotators = annotator_str.split(',')
      row = [annotator_to_most_common_label[ann] for ann in annotators]
      targets_most_freq.append(row)

  # Final output
  return(targets_most_freq)

c = list()
e = list()
n = list()
for k in range(len(targets_pe_ven_train)):
  c.append(targets_pe_ven_train[k][0])
  e.append(targets_pe_ven_train[k][1])
  n.append(targets_pe_ven_train[k][2])

# Combine into a list of [c, e, n] matrices
label_matrices = [
    most_frequent_pe(annotators_pe_ven_train,c,annotators_pe_ven_dev),  # contradiction
    most_frequent_pe(annotators_pe_ven_train,e,annotators_pe_ven_dev),  # entailment
    most_frequent_pe(annotators_pe_ven_train,n,annotators_pe_ven_dev)   # neutral
]

# Transpose to get format: one matrix per example (columns)
predictions_mostfreq_pe_dev_ven = list(map(list, zip(*label_matrices)))

print('Most frequent perspectivist evaluation score for VariErrNLI dataset: ' +str(perspectivist_evaluation('VariErrNLI',targets_pe_ven_dev,predictions_mostfreq_pe_dev_ven)))

df = pd.DataFrame()
df['id'] = ids_ven_dev
df['soft_pred']= predictions_mostfreq_soft_ven
df.to_csv('/home/irs38/varierr/results_mostfrequent_soft/ven_dev_soft.tsv', sep ='\t',header= False, index= False)

MODEL_NAME = "microsoft/deberta-v3-large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VariErrNLIDataset(Dataset):
    def __init__(self, data_dict, targets_soft, annotations_possible):
        self.data = list(data_dict.values())
        self.targets = targets_soft
        self.annotations_possible = annotations_possible
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        soft = self.targets[idx]         # [3 × 2]
        possible = self.annotations_possible[idx]  # [3], e.g. [1, 1, 0]

        # Input pair: premise + hypothesis
        text = item["text"]['context'] + " [SEP] " + item["text"]['statement']
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(soft, dtype=torch.float32),  # [3, 2]
            "possible": torch.tensor(possible, dtype=torch.float32)  # [3]
        }


class SoftLabelClassifier(nn.Module):
    def __init__(self, num_labels_per_class=2, num_classes=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.encoder.config.hidden_size
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, num_labels_per_class) for _ in range(num_classes)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state 
        cls_output = last_hidden_state[:, 0, :] 
        #pooled = outputs.pooler_output  # shape: (batch_size, hidden_size)
        logits = [classifier(cls_output) for classifier in self.classifiers]  # list of 3 tensors
        probs = [torch.softmax(logit, dim=-1) for logit in logits]
        return torch.stack(probs, dim=1)  # shape: (batch_size, 3, 2)

def evaluate(model, dataloader, targets):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)  # shape: (batch, 3, 2)
            all_preds.extend(outputs.cpu().numpy())
    
    return multilabel_average_MD(targets, all_preds)

def masked_kl_div_loss(log_preds, targets, mask):
    """
    log_preds: [batch_size, 3, 2] (log probabilities from log_softmax)
    targets: [batch_size, 3, 2] (true distributions)
    mask: [batch_size, 3] (1 = valid label, 0 = ignore this label)
    """
    # KL divergence per binary label
    kl_div = F.kl_div(log_preds, targets, reduction='none')  # [batch, 3, 2]

    # Sum over binary choices (2)
    kl_per_label = kl_div.sum(dim=-1)  # [batch, 3]

    # Apply the mask
    masked_kl = kl_per_label * mask  # [batch, 3]

    # Normalize over number of valid labels (to prevent biasing low-label-count examples)
    valid_counts = mask.sum(dim=1)  # [batch]
    loss = masked_kl.sum(dim=1) / (valid_counts + 1e-8)  # [batch]

    return loss.mean()


def train_with_early_stopping(model, train_loader, val_loader, val_targets, optimizer, 
                              max_epochs=20, patience=3):
    best_score = float('inf')  # for MD, lower is better
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)          # [batch, 3, 2]
            possible = batch["possible"].to(device)      # [batch, 3]

            # Compute masked KL divergence loss
            outputs = model(input_ids, attention_mask)  # shape [batch_size, 3, 2]
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(outputs, labels)
            weights = (labels.max(dim=-1).values - 0.5).clamp(min=0)
            mask = weights.unsqueeze(-1)  # shape [batch_size, 3, 1]
            loss = ((outputs - labels)**2 * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Evaluate on validation set
        val_score = evaluate(model, val_loader, val_targets)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val MD: {val_score:.4f}")

        # Early stopping logic
        if val_score < best_score:
            best_score = val_score
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Stopping early after {epoch+1} epochs.")
            break

    # Load the best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


# Prepare dataset and dataloader
train_dataset = VariErrNLIDataset(dataVEN_train, targets_soft_ven_train, annotations_possible_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Model, optimizer
model = SoftLabelClassifier().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

val_dataset = VariErrNLIDataset(dataVEN, targets_soft_ven_dev, annotations_possible_dev)
val_loader = DataLoader(val_dataset, batch_size=16)

# Train
model = train_with_early_stopping(
    model, train_loader, val_loader, targets_soft_ven_dev,
    optimizer, max_epochs=20, patience=3
)

score_soft = evaluate(model, val_loader, targets_soft_ven_dev)
print("Soft score (MD):", score_soft)


