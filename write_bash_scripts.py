from random import shuffle as shuffled

with open(f"sweep4.sh", "w") as f:
    f.write("#!/bin/bash\n")

commands = []

script = "python3 solution.py --entailment_model finetuned_explanations"
for model in ["combined_fusion", "combined_fusion_MLP"]: #"linear", "combined", 
    for dropout_value in [0.1, 0.2]:
        for loss_fn in ["cross_entropy", "cross_label"]:
            for learning_rate in [1e-5, 1e-4]: #1e-2, 1e-3, 1e-4, 
                weight_decay = learning_rate / 10
                for combinations in [""]:#, "--combinations"]:
                    for entpen in ["", "--entropy_penalty --beta 0.05", "--entropy_penalty --beta 0.1"]:
                        for temperature in ["", "--temperature_annealing --temperature 2", "--temperature_annealing --temperature 1.5"]: 
                            for reg in ["", "--regularise_against_mean_distribution"]:
                                for sum in ["", "--sum_lower_than_one_penalty"]:
                                    for n_unfreeze in ["--n_unfreeze 0", "--n_unfreeze 3", "--n_unfreeze 2"]:
                                        for class_weights in [""]: #, "--class_weights"
                                            command = f"{script} --model {model} --dropout_value {dropout_value} --loss_fn {loss_fn} --learning_rate {learning_rate} --weight_decay {weight_decay} {combinations} {entpen} {temperature} {reg} {sum} {n_unfreeze} {class_weights}\n"
                                            commands.append(command)
# shuffle commands
shuffled(commands)

for command in commands:                             
    with open(f"sweep4.sh", "a") as f:
        f.write(command)

