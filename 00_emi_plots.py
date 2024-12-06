import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.lines import Line2D

matplotlib.rcParams.update({'font.size': 18})
plt.rc("axes.spines", top=False, right=True)
matplotlib.rcParams['lines.markersize'] = 10
matplotlib.rcParams['lines.linewidth'] = 4

colours = ['#006BA4', '#FF800E', '#ABABAB', '#FDE725', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
colours1 = ['#440154', '#482878', '#3E4A89', '#2C6A8F', '#21908D', '#27AD81', '#5DC863', '#FDE725']
data_path = Path('data/')
output_path = Path('output/plots_emi_py')

def format_y_ticks(value, _):
  return f'{value:.1f}'

#Fig. 1

blue = "#0015BC"
red = "#FF0000"
#set up panels
fig = plt.figure(figsize=(19, 20))
gs = gridspec.GridSpec(4, 1, )

# Main Plot
ax_main = plt.subplot(gs[0, 0])
ax_main.grid(True, axis='x')
ax_main.set_title('A', fontweight='bold', loc='left', x=0.01, y=0.9)

ax_party = plt.subplot(gs[1, 0])
ax_party.grid(True, axis='x')
ax_party.set_title('B', fontweight='bold', loc='left', x=0.01, y=0.9)

ax_pol = plt.subplot(gs[2, 0])
ax_pol.grid(True, axis='x')
ax_pol.set_title('C', fontweight='bold', loc='left', x=0.01, y=0.9)

ax_prod = plt.subplot(gs[3, 0])
ax_prod.grid(True, axis='x')
ax_prod.set_title('D', fontweight='bold', loc='left', x=0.01, y=0.9)

emi_df = pd.read_csv(Path(data_path, 'congress_EMI_w2v_bootstrap_CIs.csv'))
emi_party = pd.read_csv(Path(data_path, 'congress_EMI_party_w2v_bootstrap_CIs.csv'))
emi_prod = pd.read_csv(Path(data_path, 'emi_congressw2v_prod_variables_public_laws.csv'))

col = 'evidence_minus_intuition_score'
label = 'EMI'
ax_main.plot(emi_df['year'], emi_df[f'{col}'], color='black',
             marker='o',
             linewidth=0.0,
             linestyle='',
            )
ax_main.fill_between(x='year', y1='lower_bound', y2='upper_bound', data=emi_df,  alpha=0.4, color='black')
ax_main.set_ylabel(label, fontweight='bold')

for party, color in zip(["Democrat", "Republican"], [blue, red]):
    df = emi_party[(emi_party['party'] == party)]                                     
    marker = 'o' if party == "Democrat" else 's'
    party_label = 'Democrats' if party == 'Democrat' else 'Republicans'
    ax_party.plot(df['starting_year'], df[f'{col}'], color=color, label=party_label,
                  marker=marker,
                 linewidth=0.0,
                 )
    ax_party.fill_between(x='starting_year', y1='lower_bound', y2='upper_bound', data=df,  alpha=0.4, color=color)

handler1 = Line2D([0], [0], marker='o', color='blue', label=' Democrats')
handler2 = Line2D([0], [0], marker='s', color='red', label=' Republicans')

ax_party.legend(handles=[handler1,handler2], loc='lower left', 
                borderaxespad=-0.09,
                bbox_to_anchor=(0.08, 0.02), frameon=False, 
                handlelength=1.0, labelspacing=0.5, handletextpad=0.1)

ax_party.set_ylabel('EMI', fontweight='bold')

prod_axes = [ax_prod, ax_prod.twinx()]
label=' MLI'
prod_axes[0].plot(emi_prod['starting_year'], 
                 emi_prod['MLI'],
                 color=colours1[0], label=label,
                  marker='o',                 
                 )
prod_axes[0].set_ylabel('MLI', fontweight='bold')#Major Legislation Index

label=' Public Laws'
prod_axes[1].bar(emi_prod['starting_year'], emi_prod['nlaw'], color=colours[6],
                 alpha=0.5,
                 label=label,
                 )

prod_axes[1].set_ylabel(label, fontweight='bold')
ax_prod.legend(loc='lower left', borderaxespad=-0.1, 
           bbox_to_anchor=(0.005, 0.7), frameon=False, handlelength=1.0, labelspacing=0.5, handletextpad=0.1)
prod_axes[1].legend(loc='lower left', borderaxespad=-0.1, 
           bbox_to_anchor=(0.005, 0.6), frameon=False, handlelength=1.0, labelspacing=0.5, handletextpad=0.1)
polineq_axes = [ax_pol, ax_pol.twinx()]

label = ' Inequality'
polineq_axes[0].plot(emi_prod['starting_year'], 
                 emi_prod['share_ptinc_top1pct'],
                 color=colours[1], label=label, marker='o',
                 )
polineq_axes[0].set_ylabel(label, fontweight='bold')

polineq_axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5))

label=' Polarization'
polineq_axes[1].plot(emi_prod['starting_year'], emi_prod['Avg_pol'], 
                     color=colours[4], label=label, marker='s',
                    )

polineq_axes[1].set_ylabel(label, fontweight='bold')
polineq_axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
polineq_lines = []
for ax in polineq_axes:
    line, label = ax.get_legend_handles_labels()
    polineq_lines +=line

ax_pol.legend(handles= polineq_lines, loc='lower left', borderaxespad=-0.1, 
           bbox_to_anchor=(0.03, 0.03), frameon=False, handlelength=1.0, labelspacing=0.5, handletextpad=0.1)

ax_prod.set_xlabel('Starting year of Congressional session', fontweight='bold')

plt.subplots_adjust(wspace=0.01)
plt.tight_layout()
plt.savefig(Path(output_path, 'US_EMI_party_prod_polineq_congress_w2v_1col.pdf'), format='pdf', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_party_prod_polineq_congress_w2v_1col.png'), format='png', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_party_prod_polineq_congress_w2v_1col.svg'), format='svg', dpi=300)

#Fig. 2
matplotlib.rcParams.update({'font.size': 18})
blue = "#0015BC"
red = "#FF0000"
#set up panels
fig, axes = plt.subplots(2, 1, figsize=(19, 20), sharex=True)
plt.rc("axes.spines", top=False, right=True)
for ax in axes:
    ax.grid(True, axis='x')
col = 'evidence_minus_intuition_score'
w2v_scores_party_chamber = pd.read_csv(Path(data_path, 'congress_EMI_party_chamber_w2v_bootstrap_CIs.csv'))
for chamber, ax in zip(['S', 'H'], axes):
    if chamber == 'S':
        title = 'US Senate' 
    elif chamber == 'H':
        title = 'US House' 
    ax.set_title(title, fontweight='bold')
    w2v_scores = w2v_scores_party_chamber[(w2v_scores_party_chamber['chamber'] == chamber)]
    handles = []
    for party, color in zip(["Democrat", "Republican"], [blue, red]):
        marker = 'o' if party == "Democrat" else 's'
        party_label = ' Democrats' if party == 'Democrat' else ' Republicans'
        df = w2v_scores[(w2v_scores['party'] == party)]
                                          
        ax.set_ylabel('EMI', fontweight='bold')
        ax.plot(df['starting_year'], df[f'{col}'], color=color, label=party_label, marker=marker, linewidth=0.0)
        ax.fill_between(x='starting_year', y1='lower_bound', y2='upper_bound', data=df,  alpha=0.4, color=color)

        handles.append(Line2D([0], [0], marker=marker, color=color, label=party_label))
    
    ax.legend(handles=handles, loc='lower left', 
                    borderaxespad=-0.09,
                    bbox_to_anchor=(0.08, 0.02), frameon=False, 
                    handlelength=1.0, labelspacing=0.5, handletextpad=0.1)
    
ax.set_xlabel('Starting year of Congressional session', fontweight='bold')
plt.tight_layout()
plt.savefig(Path(output_path, 'US_EMI_party_chamber_congress_w2v.pdf'), format='pdf', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_party_chamber_congress_w2v.png'), format='png', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_party_chamber_congress_w2v.svg'), format='svg', dpi=300)

#component scores
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import matplotlib
matplotlib.rcParams['lines.markersize'] = 12
matplotlib.rcParams['lines.linewidth'] = 1
# Set up the figure and grid
fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(1, 1)

# Main Plot
ax = plt.subplot(gs[0, 0])
ax.grid(True, axis='x')
emi_year_avg = pd.read_csv(Path(data_path, 'uscongress_emi_w2v.csv'))
sns.lineplot(x='starting_year', y='evidence_z', data=emi_year_avg,
             marker='.', color='blue', label='Evidence', ax=ax)

sns.lineplot(x='starting_year', y='intuition_z', data=emi_year_avg,
             marker='.', color='orange', label='Intuition', ax=ax)

ax.set_ylabel('Average score', fontweight='bold')
ax.set_xlabel('Starting year', fontweight='bold')
ax.legend(loc='lower left', frameon=False)

plt.tight_layout()
plt.savefig(Path(output_path, 'US_EMI_components.pdf'), format='pdf', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_components.png'), format='png', dpi=300)
plt.savefig(Path(output_path, 'US_EMI_components.svg'), format='svg', dpi=300)
