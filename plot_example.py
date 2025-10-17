from matplotlib_venn import venn3
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Entropy values
entropy_values = {
    '100': 0.19,      # Contradiction only
    '010': 0.12,      # Entailment only
    '001': -0.13,     # Neutral only
    '110': 0.27,      # Contradiction ∩ Entailment
    '101': -0.096,    # Contradiction ∩ Neutral
    '011': -0.086,    # Entailment ∩ Neutral
    '111': 0.069      # All three
}

plt.figure(figsize=(9, 9))
v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1),
          set_labels=('Contradiction', 'Entailment', 'Neutral'))

# --- Colormap setup: dark red → peach → bright yellow ---
vals = np.array(list(entropy_values.values()))
norm = mcolors.TwoSlopeNorm(vmin=vals.min(), vcenter=0, vmax=vals.max())
cmap = mcolors.LinearSegmentedColormap.from_list(
    "entropy_peachyellow", ["darkred", "peachpuff", "yellow"]
)

# --- Fill regions with color by entropy ---
for region_id, val in entropy_values.items():
    patch = v.get_patch_by_id(region_id)
    if patch:
        patch.set_facecolor(cmap(norm(val)))
        patch.set_edgecolor('black')
        patch.set_linewidth(2.2)

# --- Add entropy text labels inside regions ---
for region_id, val in entropy_values.items():
    label = v.get_label_by_id(region_id)
    if label:
        label.set_text(f"{val:.2f}")
        label.set_fontsize(18)
        label.set_fontweight('bold')

# --- Make main labels (set names) large and bold ---
for lbl in v.set_labels:
    if lbl:
        lbl.set_fontsize(22)
        lbl.set_fontweight('bold')

plt.tight_layout()


plt.savefig("semantic_entropy_heatmap.png")