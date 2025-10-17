from datasets import load_dataset
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo, EntailmentGPT35, EntailmentDeberta
from torch.nn.functional import log_softmax
from collections import Counter
import numpy as np
import torch.nn.functional as F

from huggingface_hub import login

num_return = 32

from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

from transformers import StoppingCriteria, StoppingCriteriaList
import torch

class StopAtSentenceEnd(StoppingCriteria):
    def __init__(self, tokenizer, start_len):
        self.tokenizer = tokenizer
        self.start_len = start_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Called every step of generation. Return True to stop.
        """
        # Only look at new tokens
        new_tokens = input_ids[0, self.start_len:]

        # Decode the new part only (can be slow but reliable)
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Simple heuristic: stop at sentence-ending punctuation
        if decoded.endswith(('.', '!', '?', "\n\n")):  # You can customize this
            return True

        return False
    
ds = load_dataset("mainlp/varierr")

print(ds["train"][2])
print(f"Size of the dataset: {len(ds['train'])}")
print(f"Number of items with has_ambiguity set to True: {sum([1 for item in ds['train'] if item['has_ambiguity']])}")

semantic_ids_annotators = []
mean_semantic_entropy = []
mean_semantic_entropy_e = []
mean_semantic_entropy_n = []
mean_semantic_entropy_c = []
ambiguous_entailment = []
unambiguous_entailment = []
unambiguous_mean_semantic_entropy = []
ambiguous_mean_semantic_entropy = []
has_two_different_annotations_by_same_annotator = []

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


for item in ds['train']:
    example_e = (
        "Statement: Everything can be found inside a shopping mall.\n"
        "Context: Enter the realm of shopping malls, where everything you're looking for is available without moving your car.\n"
        "Judgment: Entailment\n"
        "Explanation: The context implies that the shopping mall has everything one might look for, as it can be found without moving your car.\n\n"

        "Statement: The matter of whether or not the Mass is a sacrifice for the remission of sins is controversial.\n"
        "Context: As for the divisive issue of whether the Mass is a sacrifice for the remission of sins, the statement affirms that Christ's death upon the cross ...\n"
        "Judgment: Entailment\n"
        "Explanation: The context states that the Mass being a sacrifice for the remission of sins is divisive, which can be interpreted as a synonym for controversial.\n\n"

        # "Statement: \n"
        # "Context: \n"
        # "Judgment: \n"
        # "Explanation: \n\n"
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

        # "Statement: \n"
        # "Context: \n"
        # "Judgment: \n"
        # "Explanation: \n\n"
    )
    example_c = (
        "Statement: He had recently seen pictures depicting those things.\n"
        "Context: He hadn't seen even pictures of such things since the few silent movies run in some of the little art theaters.\n"
        "Judgment: Contradiction\n"
        "Explanation: If the pronoun 'he' and the object 'those things' refer to the same things in the statement and the context, then the statement negates the context.\n\n"

        "Statement: Octavius Decatur Gass refers to four people.\n"
        "Context: One opportunist who stayed was Octavius Decatur Gass.\n"
        "Judgment: Contradiction\n"
        "Explanation: The context names one person as Octavius Decatur Gass, and does not mention additional referrents.\n\n"
        
        # "Statement: \n"
        # "Context: \n"
        # "Judgment: \n"
        # "Explanation: \n\n"
    )
    examples = example_e + example_n + example_c

    input_e = example_e + f"Statement: {item['statement']}\nContext: {item['context']}\nJudgment: Entailment\nExplain why this is the most likely judgment label."
    input_n = example_n + f"Statement: {item['statement']}\nContext: {item['context']}\nJudgment: Neutral\nExplain why this is the most likely judgment label."
    input_c = example_c + f"Statement: {item['statement']}\nContext: {item['context']}\nJudgment: Contradiction\nExplain why this is the most likely judgment label."
    input_joined = examples + f"Statement: {item['statement']}\nContext: {item['context']}\nJudgment: "
    input_ids_e = tokenizer(input_e, return_tensors="pt").to("cuda")
    input_ids_n = tokenizer(input_n, return_tensors="pt").to("cuda")
    input_ids_c = tokenizer(input_c, return_tensors="pt").to("cuda")
    input_ids_joined = tokenizer(input_joined, return_tensors="pt").to("cuda")

    input_length_e = input_ids_e["input_ids"].shape[1] 
    input_length_n = input_ids_n["input_ids"].shape[1]
    input_length_c = input_ids_c["input_ids"].shape[1]
    input_length_joined = input_ids_joined["input_ids"].shape[1]

    stopping_criteria_e = StoppingCriteriaList([StopAtSentenceEnd(tokenizer, input_length_e)])
    stopping_criteria_n = StoppingCriteriaList([StopAtSentenceEnd(tokenizer, input_length_n)])
    stopping_criteria_c = StoppingCriteriaList([StopAtSentenceEnd(tokenizer, input_length_c)])
    stopping_criteria_joined = StoppingCriteriaList([StopAtSentenceEnd(tokenizer, input_length_joined)])

    outputs_e = model.generate(**input_ids_e, max_new_tokens=100, num_return_sequences=num_return, do_sample=True, epsilon_cutoff=0.2,output_scores=True,return_dict_in_generate=True,stopping_criteria=stopping_criteria_e,pad_token_id=tokenizer.eos_token_id)
    outputs_n = model.generate(**input_ids_n, max_new_tokens=100, num_return_sequences=num_return, do_sample=True, epsilon_cutoff=0.2,output_scores=True,return_dict_in_generate=True,stopping_criteria=stopping_criteria_n,pad_token_id=tokenizer.eos_token_id)
    outputs_c = model.generate(**input_ids_c, max_new_tokens=100, num_return_sequences=num_return, do_sample=True, epsilon_cutoff=0.2,output_scores=True,return_dict_in_generate=True,stopping_criteria=stopping_criteria_c,pad_token_id=tokenizer.eos_token_id)
    outputs_joined = model.generate(**input_ids_joined, max_new_tokens=200, num_return_sequences=num_return, do_sample=True, epsilon_cutoff=0.2,output_scores=True,return_dict_in_generate=True,stopping_criteria=stopping_criteria_joined,pad_token_id=tokenizer.eos_token_id)

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
        samples_e.append(decoded)
    samples_n = []
    for sample in sequences_n:
        decoded = tokenizer.decode(sample[input_length_n:], skip_special_tokens=True)
        samples_n.append(decoded)
    samples_c = []
    for sample in sequences_c:
        decoded = tokenizer.decode(sample[input_length_c:], skip_special_tokens=True)
        samples_c.append(decoded)
    samples_joined = []
    for sample in sequences_joined:
        decoded = tokenizer.decode(sample[input_length_joined:], skip_special_tokens=True)
        samples_joined.append(decoded)

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
    example_e['question'] = item['statement']
    example_e['context'] = item['context']
    responses_n = unique_samples_n
    log_liks_n = unique_logprobs_n
    log_liks_agg_n = [np.mean(log_lik) for log_lik in log_liks_n]
    example_n = {}
    example_n['question'] = item['statement']
    example_n['context'] = item['context']
    responses_c = unique_samples_c
    log_liks_c = unique_logprobs_c
    log_liks_agg_c = [np.mean(log_lik) for log_lik in log_liks_c]
    example_c = {}
    example_c['question'] = item['statement']
    example_c['context'] = item['context']
    responses_joined = unique_samples_joined
    log_liks_joined = unique_logprobs_joined
    log_liks_agg_joined = [np.mean(log_lik) for log_lik in log_liks_joined]
    example_joined = {}
    example_joined['question'] = item['statement']
    example_joined['context'] = item['context']

    num_per_label = {"Entailment": 0, "Neutral": 0, "Contradiction": 0}
    responses_joined_ordered = []
    log_liks_joined_ordered = []
    for label in ["Entailment", "Neutral", "Contradiction"]:
        for i,response in enumerate(responses_joined):
            if response[0].strip().startswith(label):
                num_per_label[label] += 1
                response_no_label = response[0].strip()[len(label):].strip()
                responses_joined_ordered.append(response_no_label)
                log_liks_joined_ordered.append(log_liks_joined[i])
    
    entailment_model = EntailmentDeberta()

    responses_gold = []
    log_liks_gold = []
    num_per_label_gold = {"Entailment": 0, "Neutral": 0, "Contradiction": 0}
    for e in item["entailment"]:
        num_per_label_gold["Entailment"] += 1
        responses_gold.append(e["reason"])
    for n in item["neutral"]:
        num_per_label_gold["Neutral"] += 1
        responses_gold.append(n["reason"])
    for c in item["contradiction"]:
        num_per_label_gold["Contradiction"] += 1
        responses_gold.append(c["reason"])

    log_liks_gold = []
    for sentence in responses_gold:
        sentence_ids = tokenizer(sentence, return_tensors='pt').to("cuda")
        input_ids = torch.cat([input_ids_joined["input_ids"], sentence_ids["input_ids"]], dim=1)
        labels = torch.cat([torch.full_like(input_ids_joined["input_ids"], -100),sentence_ids["input_ids"]], dim=1)
        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = sentence_ids["input_ids"][:, 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        total_logprob = target_log_probs.sum(dim=-1).to(torch.float32).cpu().detach().numpy().item()        
        log_liks_gold.append(total_logprob)

    semantic_ids_all = get_semantic_ids(responses_e + responses_n + responses_c, model=entailment_model,strict_entailment=False, example=item)
    semantic_ids_joined = get_semantic_ids(responses_joined_ordered, model=entailment_model, strict_entailment=False, example=item)
    semantic_ids_gold = get_semantic_ids(responses_gold, model=entailment_model, strict_entailment=False, example=item)

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

    renumbered_gold_e = renumber(semantic_ids_gold[:num_per_label_gold["Entailment"]])
    renumbered_gold_n = renumber(semantic_ids_gold[num_per_label_gold["Entailment"]:num_per_label_gold["Entailment"]+num_per_label_gold["Neutral"]])
    renumbered_gold_c = renumber(semantic_ids_gold[num_per_label_gold["Entailment"]+num_per_label_gold["Neutral"]:])
    renumbered_gold_en = renumber(semantic_ids_gold[:num_per_label_gold["Entailment"]+num_per_label_gold["Neutral"]])
    renumbered_gold_ec = renumber(semantic_ids_gold[:num_per_label_gold["Entailment"]]+semantic_ids_gold[num_per_label_gold["Entailment"]+num_per_label_gold["Neutral"]:num_per_label_gold["Entailment"]+num_per_label_gold["Neutral"]+num_per_label_gold["Contradiction"]])
    renumbered_gold_nc = renumber(semantic_ids_gold[num_per_label_gold["Entailment"]:num_per_label_gold["Entailment"]+num_per_label_gold["Neutral"]+num_per_label_gold["Contradiction"]])

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

    log_likelihood_per_semantic_id_gold_e = logsumexp_by_id(renumbered_gold_e, log_liks_gold[:num_per_label_gold["Entailment"]], agg='sum_normalized')
    semantic_entropy_gold_e = predictive_entropy_rao(log_likelihood_per_semantic_id_gold_e)
    log_likelihood_per_semantic_id_gold_n = logsumexp_by_id(renumbered_gold_n, log_liks_gold[num_per_label_gold["Entailment"]:num_per_label_gold["Entailment"]+num_per_label_gold["Neutral"]], agg='sum_normalized')
    semantic_entropy_gold_n = predictive_entropy_rao(log_likelihood_per_semantic_id_gold_n)
    log_likelihood_per_semantic_id_gold_c = logsumexp_by_id(renumbered_gold_c, log_liks_gold[num_per_label_gold["Entailment"]+num_per_label_gold["Neutral"]:], agg='sum_normalized')
    semantic_entropy_gold_c = predictive_entropy_rao(log_likelihood_per_semantic_id_gold_c)
    log_likelihood_per_semantic_id_gold_all = logsumexp_by_id(semantic_ids_gold, log_liks_gold, agg='sum_normalized')
    semantic_entropy_gold_all = predictive_entropy_rao(log_likelihood_per_semantic_id_gold_all)
    log_likelihood_per_semantic_id_gold_en = logsumexp_by_id(renumbered_gold_en, log_liks_gold[:num_per_label_gold["Entailment"]+num_per_label_gold["Neutral"]], agg='sum_normalized')
    semantic_entropy_gold_en = predictive_entropy_rao(log_likelihood_per_semantic_id_gold_en)
    log_likelihood_per_semantic_id_gold_ec = logsumexp_by_id(renumbered_gold_ec, log_liks_gold[:num_per_label_gold["Entailment"]]+log_liks_gold[num_per_label_gold["Entailment"]+num_per_label_gold["Neutral"]:num_per_label_gold["Entailment"]+num_per_label_gold["Neutral"]+num_per_label_gold["Contradiction"]], agg='sum_normalized')
    semantic_entropy_gold_ec = predictive_entropy_rao(log_likelihood_per_semantic_id_gold_ec)
    log_likelihood_per_semantic_id_gold_nc = logsumexp_by_id(renumbered_gold_nc, log_liks_gold[num_per_label_gold["Entailment"]:num_per_label_gold["Entailment"]+num_per_label_gold["Neutral"]+num_per_label_gold["Contradiction"]], agg='sum_normalized')
    semantic_entropy_gold_nc = predictive_entropy_rao(log_likelihood_per_semantic_id_gold_nc)

    all_labels = []
    verified_labels = []
    for e in item["entailment"]:
        all_labels.append("entailment")
        if not e["self_corrected"]:
            verified_labels.append("entailment")
    for n in item["neutral"]:
        all_labels.append("neutral")
        if not n["self_corrected"]:
            verified_labels.append("neutral")
    for c in item["contradiction"]:
        all_labels.append("contradiction")
        if not c["self_corrected"]:
            verified_labels.append("contradiction")

    generated_labels = [r[0].split("Explanation")[0] for r in responses_joined]

    if len(set(all_labels)) > 1:
        print(f"Verified labels: {verified_labels}")
        print(f"All labels: {Counter(all_labels)}")
        print(f"Item {item['id']}:")
        print(f"Statement: {item['statement']}")
        print(f"Context: {item['context']}")
        print(f"Generated labels: {Counter(generated_labels)}")
        import pdb; pdb.set_trace()
    mean_semantic_entropy_e.append(semantic_entropy_e)
    mean_semantic_entropy_n.append(semantic_entropy_n)
    mean_semantic_entropy_c.append(semantic_entropy_c)
    mean_semantic_entropy.append(semantic_entropy_all)



    entailment_reasons = [item["entailment"][i]["reason"] for i in range(len(item["entailment"]))]
    neutral_reasons = [item["neutral"][i]["reason"] for i in range(len(item["neutral"]))]
    contradiction_reasons = [item["contradiction"][i]["reason"] for i in range(len(item["contradiction"]))]

    entailment_model = EntailmentDeberta()
    semantic_ids_entailmentneutralcontradiction = get_semantic_ids(
        entailment_reasons+neutral_reasons+contradiction_reasons, model=entailment_model,
        strict_entailment=False, example=item)
    if item["has_ambiguity"]:
        ambiguous_entailment.append(semantic_ids_entailmentneutralcontradiction)
    else:
        unambiguous_entailment.append(semantic_ids_entailmentneutralcontradiction)

    reasons_by_annotator = {}
    for aidx,annotation in enumerate(item["entailment"]):
        if not item["entailment"][aidx]["self_corrected"]:
            annotator = annotation["annotator"]
            if annotator not in reasons_by_annotator:
                reasons_by_annotator[annotator] = []
            reasons_by_annotator[annotator].append(annotation["reason"])
    for aidx,annotation in enumerate(item["neutral"]):
        if not item["neutral"][aidx]["self_corrected"]:
            annotator = annotation["annotator"]
            if annotator not in reasons_by_annotator:
                reasons_by_annotator[annotator] = []
            reasons_by_annotator[annotator].append(annotation["reason"])
    for aidx,annotation in enumerate(item["contradiction"]):
        if not item["contradiction"][aidx]["self_corrected"]:
            annotator = annotation["annotator"]
            if annotator not in reasons_by_annotator:
                reasons_by_annotator[annotator] = []
            reasons_by_annotator[annotator].append(annotation["reason"])

    semantic_ids_per_annotator = {}
    for annotator, reasons in reasons_by_annotator.items():
        semantic_ids = get_semantic_ids(reasons, model=entailment_model, strict_entailment=False, example=item)
        semantic_ids_per_annotator[annotator] = semantic_ids
        if len(semantic_ids) > 1:
            semantic_ids_annotators.append(semantic_ids_per_annotator[annotator])

    if any([len(sem)>1 for sem in semantic_ids_per_annotator.values()]):
        has_two_different_annotations_by_same_annotator.append(1)
    else:
        has_two_different_annotations_by_same_annotator.append(0)

ambiguous_items_ids = [i for i,item in enumerate(ds['train']) if item['has_ambiguity']]
mismatch_count = [i for i in range(len(ds["train"])) if (i not in ambiguous_items_ids and has_two_different_annotations_by_same_annotator[i]==1) or (i in ambiguous_items_ids and has_two_different_annotations_by_same_annotator[i]==0)]

import pdb; pdb.set_trace()

print("mean number of semantic ids for ambiguous items:", np.mean([len(set(ids)) for ids in ambiguous_entailment]))
print("mean number of semantic ids for unambiguous items:", np.mean([len(set(ids)) for ids in unambiguous_entailment]))
print("mean number of semantic ids per annotator:", np.mean([len(set(ids)) for ids in semantic_ids_annotators]))

print("mean semantic entropy entailment:", np.nanmean(mean_semantic_entropy_e))
print("mean semantic entropy neutral:", np.nanmean(mean_semantic_entropy_n))
print("mean semantic entropy contradiction:", np.nanmean(mean_semantic_entropy_c))
print("mean semantic entropy all:", np.nanmean(mean_semantic_entropy))

print("mean semantic entropy entailment for ambiguous items:", np.nanmean([mean_semantic_entropy_e[i] for i in ambiguous_items_ids]))
print("mean semantic entropy neutral for ambiguous items:", np.nanmean([mean_semantic_entropy_n[i] for i in ambiguous_items_ids]))
print("mean semantic entropy contradiction for ambiguous items:", np.nanmean([mean_semantic_entropy_c[i] for i in ambiguous_items_ids]))
print("mean semantic entropy all for ambiguous items:", np.nanmean([mean_semantic_entropy[i] for i in ambiguous_items_ids]))

print("mean semantic entropy entailment for non-ambiguous items:", np.nanmean([mean_semantic_entropy_e[i] for i in range(len(ds['train'])) if i not in ambiguous_items_ids]))
print("mean semantic entropy neutral for non-ambiguous items:", np.nanmean([mean_semantic_entropy_n[i] for i in range(len(ds['train'])) if i not in ambiguous_items_ids]))
print("mean semantic entropy contradiction for non-ambiguous items:", np.nanmean([mean_semantic_entropy_c[i] for i in range(len(ds['train'])) if i not in ambiguous_items_ids]))
print("mean semantic entropy all for non-ambiguous items:", np.nanmean([mean_semantic_entropy[i] for i in range(len(ds['train'])) if i not in ambiguous_items_ids]))


import pdb; pdb.set_trace()
