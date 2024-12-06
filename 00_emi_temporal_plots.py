import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import matplotlib
plt.rc("axes.spines", top=False, right=True)
matplotlib.rcParams['lines.markersize'] = 12
matplotlib.rcParams['lines.linewidth'] = 1
fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(1, 1, )

data_path = Path('data/')
output_path = Path('output/plots_emi_py')
df_emi_2decades = pd.read_csv(Path(data_path, 'uscongress_emi_2decades_w2v_downsampled.csv'))

ax = plt.subplot(gs[0, 0])
ax.grid(True, axis='x')
sns.lineplot(x='starting_year', y='evidence_minus_intuition_score', data=df_emi_2decades,
             marker='.', ax=ax, color='black'
            )
ax.set_ylabel('EMI', fontweight='bold')
ax.set_xlabel('Starting year', fontweight='bold')
plt.tight_layout()
plt.savefig(Path(output_path, 'US_EMI_2decades_downsampled.pdf'), format='pdf', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_2decades_downsampled.png'), format='png', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_2decades_downsampled.svg'), format='svg', dpi=300)
plt.close()

#plot pairwise similarity
matplotlib.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 8))
def visualize_two_similarity_matrices(matrix1, matrix2, title1="Matrix 1", title2="Matrix 2", output_file=None):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)  # Two panels side by side

    # Mask the upper triangle
    mask1 = np.triu(np.ones_like(matrix1, dtype=bool))
    mask2 = np.triu(np.ones_like(matrix2, dtype=bool))

    # Plot the first matrix
    sns.heatmap(matrix1, annot=True, fmt='.2f', cmap="cividis", vmin=0, vmax=1, square=True, cbar_kws={"shrink": .7},
                mask=mask1, annot_kws={"fontsize": 12}, ax=axes[0])
    axes[0].set_title(title1)
    axes[0].set_xlabel("Decades starting", fontweight='bold')
    axes[0].set_ylabel("Decades starting", fontweight='bold')

    # Plot the second matrix
    sns.heatmap(matrix2, annot=True, fmt='.2f', cmap="cividis", vmin=0, vmax=1, square=True, cbar_kws={"shrink": .7},
                mask=mask2, annot_kws={"fontsize": 12}, ax=axes[1])
    axes[1].set_title(title2)
    axes[1].set_xlabel("Decades starting", fontweight='bold')
    axes[1].set_ylabel("Decades starting", fontweight='bold')
    plt.savefig(Path(output_path, f'{output_file}.pdf'), format='pdf', dpi=300)
    plt.savefig(Path(output_path, f'{output_file}.png'), format='png', dpi=300)
    plt.savefig(Path(output_path, f'{output_file}.svg'), format='svg', dpi=300)
    plt.close()

evidence_matrix_2decades_downsampled = pd.read_csv(Path(data_path, 'evidence_pairwise_sim_temporal.csv'), index_col=0)
evidence_matrix_2decades_randompair_downsampled = pd.read_csv(Path(data_path, 'evidence_randompair_sim_temporal.csv'), index_col=0)
visualize_two_similarity_matrices(matrix1=evidence_matrix_2decades_downsampled,
                                  matrix2=evidence_matrix_2decades_randompair_downsampled,
                                  title1='A) Evidence dictionary',
                                  title2='B) Evidence dictionary paired with random words',
                                  output_file='pairwise_sim_matrix_evidence'
                                 )
intuition_matrix_2decades_downsampled = pd.read_csv(Path(data_path, 'intuition_pairwise_sim_temporal.csv'), index_col=0)
intuition_matrix_2decades_downsampled_randompair = pd.read_csv(Path(data_path, 'intuition_randompair_sim_temporal.csv'), index_col=0)
visualize_two_similarity_matrices(matrix1=intuition_matrix_2decades_downsampled,
                                  matrix2=intuition_matrix_2decades_downsampled_randompair,
                                  title1='A) Intuition dictionary',
                                  title2='B) Intuition dictionary paired with random words',
                                  output_file='pairwise_sim_matrix_intuition'
                                 )

