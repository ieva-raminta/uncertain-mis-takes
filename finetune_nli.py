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
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from itertools import combinations
import matplotlib.pyplot as plt

ds = load_dataset("mainlp/varierr")

print(ds["train"][2])
print(f"Size of the dataset: {len(ds['train'])}")
print(f"Number of items with has_ambiguity set to True: {sum([1 for item in ds['train'] if item['has_ambiguity']])}")

nli_soft_train = []
nli_soft_dev = []

for itemid, item in enumerate(ds['train'].shuffle(seed=42)):

    annots = []
    for label in ["entailment", "neutral", "contradiction"]:
        for annotation in item[label]:
            explanation = annotation["reason"]
            annotator = annotation["annotator"]
            judgments = annotation["judgments"]
            annots.append({"annotator": annotator, "label": label, "explanation": explanation, "judgments": judgments})
    
    # for each pair of annotations
    for annotation1, annotation2 in combinations(annots, 2):
        reason1 = annotation1["explanation"]
        reason2 = annotation2["explanation"]
        annotator1 = annotation1["annotator"]
        annotator2 = annotation2["annotator"]
        judgers1 = [j["annotator"] for j in annotation1["judgments"]]
        judgers2 = [j["annotator"] for j in annotation2["judgments"]]
        annotation1_support = []
        annotation2_support = []
        # if annotator1 in judgers1: 
        #     annotation1_support.append([a["makes_sense"] for a in annotation1["judgments"] if a["annotator"] == annotator1][0])
        if annotator2 in judgers1:
            annotation1_support.append([a["makes_sense"] for a in annotation1["judgments"] if a["annotator"] == annotator2][0])
        if annotator1 in judgers2:
            annotation2_support.append([a["makes_sense"] for a in annotation2["judgments"] if a["annotator"] == annotator1][0])
        # if annotator2 in judgers2:
        #     annotation2_support.append([a["makes_sense"] for a in annotation2["judgments"] if a["annotator"] == annotator2][0])
        if len(annotation1_support) + len(annotation2_support) != 0:
            total_support = sum(annotation1_support + annotation2_support)/len(annotation1_support + annotation2_support)
            label = torch.tensor([total_support, 0, 1 - total_support], dtype=torch.float32)
            if itemid < len(ds['train']) * 0.8:
                nli_soft_train.append({"premise": reason1, "hypothesis": reason2, "label": label})
            else:
                nli_soft_dev.append({"premise": reason1, "hypothesis": reason2, "label": label})

class SoftContradictionDeberta(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/bart-large-mnli"
        )
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            "facebook/bart-large-mnli",
            num_labels=3,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, 3]
        return logits

def kl_loss_fn(log_probs, target_probs):
    """
    log_probs: [batch_size, 3], log probabilities (log Q)
    target_probs: [batch_size, 3], soft label distributions (P)
    """
    return F.kl_div(log_probs, target_probs, reduction='batchmean')

loss_fn = kl_loss_fn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SoftContradictionDeberta().to(DEVICE)
tokenizer = model.tokenizer

for name, param in model.named_parameters():
    if 'classification' in name:
        param.requires_grad = True 
    else:
        param.requires_grad = False 

def preprocess(premise, hypothesis, label):
    encoded = tokenizer(premise, hypothesis, truncation=True, padding='max_length', max_length=256, return_tensors="pt")
    return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
        "label": label.clone().detach()
    }

class NliSoftDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer):
        self.data = [preprocess(e['premise'], e['hypothesis'], e['label']) for e in examples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = NliSoftDataset(nli_soft_train, tokenizer)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

dev_dataset = NliSoftDataset(nli_soft_dev, model.tokenizer)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=16)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

plt.hist([x['label'] for x in nli_soft_train], bins=20)
plt.savefig("nli_soft_train_distribution.png")

best_val_loss = float('inf')
patience = 3
patience_counter = 0
num_epochs = 20

def evaluate(model, dev_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = loss_fn(log_probs, labels)

            total_loss += loss.item() * input_ids.size(0)
            total_examples += input_ids.size(0)

    return total_loss / total_examples


for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0

    for batch in loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask) 
        log_probs = F.log_softmax(logits, dim=-1)  # input to KL
        loss = loss_fn(log_probs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
    scheduler.step(total_train_loss / len(loader))

    avg_train_loss = total_train_loss / len(loader)
    val_loss = evaluate(model, dev_loader, loss_fn)

    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {val_loss:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")  # Save best model
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


model.load_state_dict(torch.load("best_model.pt"))
model.eval()


