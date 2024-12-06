import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
plt.rc("axes.spines", top=False, right=False)

output_path = Path('output/plots_emi_py')
data_path = Path('data/')
grouped = pd.read_csv(Path(data_path, 'us_congress_emi_topic_agg_ci.csv'))

grouped['ci_error'] = (grouped['ci_upper'] - grouped['ci_lower']) / 2
grouped.label = grouped.label.apply(lambda x: x.split('_')[-1])
# Sort the scores for better visualization
grouped_sorted = grouped.sort_values('mean_score')

# Create a horizontal bar plot with error bars
plt.figure(figsize=(12, 5))
plt.barh(
    grouped_sorted.label,
    grouped_sorted['mean_score'],
    xerr=grouped_sorted['ci_error'], 
    color='black',    
)
# Add labels and title
plt.xlabel('Average EMI Score', fontweight='bold')
plt.ylabel('CAP Topic', fontweight='bold')
plt.tight_layout()
plt.savefig(Path(output_path, 'US_EMI_topics_agg_bar.pdf'), format='pdf', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_topics_agg_bar.png'), format='png', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_topics_agg_bar.svg'), format='svg', dpi=300)
plt.close()

#plot topics 1 per panel
topic_emi_normalized = pd.read_csv(Path(data_path, 'us_congress_emi_topic_normalized.csv'))
topic_emi_normalized['label'] = topic_emi_normalized.label.apply(lambda x: x.split('_')[-1])
matplotlib.rcParams.update({'font.size': 12})
plt.rc("axes.spines", top=False, right=True)
matplotlib.rcParams['lines.markersize'] = 8
matplotlib.rcParams['lines.linewidth'] = 1

fig, axes = plt.subplots(6, 4, figsize=(20, 12), sharey=True)#sharex=True,
axes = axes.flatten()

unique_topics = topic_emi_normalized['label'].unique()
for idx, topic in enumerate(unique_topics):
    data = topic_emi_normalized[topic_emi_normalized['label'] == topic]
    sns.lineplot(x='starting_year', y='emi_normalized', data=data, ax=axes[idx], marker=".", color='black')
    axes[idx].set_title(topic, fontweight='bold')
    axes[idx].set_xlabel(None)
    axes[idx].set_ylabel(None)

for idx in range(len(unique_topics), len(axes)):
    fig.delaxes(axes[idx])
axes[0].set_ylabel('EMI', fontweight='bold')
axes[4].set_ylabel('EMI', fontweight='bold')
axes[8].set_ylabel('EMI', fontweight='bold')
axes[12].set_ylabel('EMI', fontweight='bold')
axes[16].set_ylabel('EMI', fontweight='bold')
axes[20].set_ylabel('EMI', fontweight='bold')

# Adjust layout for better fitting
plt.tight_layout(rect=[0.05, 0.05, 1, 1])

plt.savefig(Path(output_path, 'US_EMI_topics_per_panel.pdf'), format='pdf', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_topics_per_panel.png'), format='png', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_topics_per_panel.svg'), format='svg', dpi=300)
plt.close()
#topics in 1 panel with macro average overlaid
matplotlib.rcParams.update({'font.size': 12})
plt.rc("axes.spines", top=False, right=False)
matplotlib.rcParams['lines.markersize'] = 8
matplotlib.rcParams['lines.linewidth'] = 1

palette = sns.color_palette("tab20", 22)
sns.set_palette(palette)
plt.figure(figsize=(12, 6))
unique_topics = topic_emi_normalized['label'].unique()

for topic in unique_topics:
    data = topic_emi_normalized[topic_emi_normalized['label'] == topic]
    sns.lineplot(
        x='starting_year',
        y='emi_normalized',
        data=data,
        label=topic,
        alpha=0.5,
        marker="."
    )

mean_data = topic_emi_normalized.groupby(['starting_year', 'label'])['emi_normalized'].mean().reset_index()
mean_data = mean_data.groupby('starting_year')['emi_normalized'].mean().reset_index()
sns.lineplot(
    x='starting_year',
    y='emi_normalized',
    data=mean_data,
    label='Macro Average',
    color='black',
    linewidth=2,
    marker="."
)

# Add labels and legend
plt.xlabel('Starting Year', fontweight='bold')
plt.ylabel('EMI', fontweight='bold')
plt.legend(title=None, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4,
          )

plt.tight_layout()
plt.savefig(Path(output_path, 'US_EMI_topics_macroavg.pdf'), format='pdf', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_topics_macroavg.png'), format='png', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_topics_macroavg.svg'), format='svg', dpi=300)
plt.close()
