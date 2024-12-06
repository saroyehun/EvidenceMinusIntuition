#pyearth installation
#python3.6 -m pip install git+https://github.com/scikit-learn-contrib/py-earth@v0.2dev

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyearth import Earth
import re
import matplotlib
matplotlib.rcParams['lines.markersize'] = 8
matplotlib.rcParams['lines.linewidth'] = 1.0

# Configure matplotlib parameters
matplotlib.rcParams.update({'font.size': 18})
plt.rc("axes.spines", top=False, right=True)
def fit_and_plot_earth(df, ax, title):
    # Fit the Earth model
    model = Earth(max_terms=2, minspan_alpha=0.0001, endspan_alpha=0.0001, enable_pruning=True)
    model.fit(df['starting_year'].values.reshape(-1, 1), df['evidence_minus_intuition_score'])
       # Predict EMI scores
    predicted_emi = model.predict(df['starting_year'].values.reshape(-1, 1))
    
    # Plot the EMI scores and the MARS fit
    ax.plot(df['starting_year'], df['evidence_minus_intuition_score'], label="EMI", marker=".", color='black')
    ax.plot(df['starting_year'], predicted_emi, label="MARS Fit", linestyle="--", color='grey')
    
    # Extract unique breakpoints (knots) from the basis functions
    knots = set()
    for bf in model.basis_:
        if 'h(' in str(bf):
            knot = re.findall(r'\d+', str(bf))
            if knot and (int(knot[0]) > 0):
                knots.add(int(knot[0]))
    
    # Mark the breakpoints on the plot
    for knot in sorted(knots):
        ax.axvline(knot, color='red', linestyle='--')  # Breakpoint lines
    
    
    ax.set_title(title)
    ax.set_ylabel("EMI", fontweight='bold')
    ax.legend()
    ax.grid(axis='x')
    

output_path = Path('output/plots_emi_py')
data_path = Path('data/')


# With macro average over topics
df_emi_2decades = pd.read_csv(Path(data_path, 'us_congress_emi_topic_avg_2decades.csv'))
# df_emi = pd.read_csv(Path(data_path, 'us_congress_emi_topic_avg.csv'))

# Set up subplots with 2 rows and 1 column
fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True, sharex=True)


# Apply the function to each dataframe and plot in corresponding subplot
fit_and_plot_earth(df_emi, axes[0], "A) EMI (Full Corpus Embeddings)")
fit_and_plot_earth(df_emi_2decades, axes[1], "B) EMI (2-Decade Embeddings)")
axes[1].set_xlabel("Starting year", fontweight='bold')


plt.savefig(Path(output_path, 'US_EMI_full_temporal_topicmacroavg_mars_fit.pdf'), format='pdf', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_full_temporal_topicmacroavg_mars_fit.png'), format='png', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_full_temporal_topicmacroavg_mars_fit.svg'), format='svg', dpi=300)

# Without macro average over topics
df_emi_2decades = pd.read_csv(Path(data_path, 'uscongress_emi_2decades_w2v_downsampled.csv'))
df_emi = pd.read_csv(Path(data_path, 'uscongress_emi_w2v.csv'))

fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, sharex=True)

fit_and_plot_earth(df_emi, axes[0], "A) EMI (Full Corpus Embeddings)")
fit_and_plot_earth(df_emi_2decades, axes[1], "B) EMI (2-Decade Embeddings)")
axes[1].set_xlabel("Starting year", fontweight='bold')
plt.savefig(Path(output_path, 'US_EMI_full_temporal_mars_fit.pdf'), format='pdf', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_full_temporal_mars_fit.png'), format='png', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_full_temporal_mars_fit.svg'), format='svg', dpi=300)

    
