import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 18})
plt.rc("axes.spines", top=True, right=True)

data_path = Path('../data/')
output_path = Path('../output/')

fig = plt.figure(figsize=(19, 8))
speech_count = pd.read_csv(Path(data_path, 'speech_count.csv'))
plt.bar(speech_count['starting_year'], speech_count['speech_count'], color='black')
plt.margins(y=0.01, x=0.01)
plt.ylabel('Number of Speeches', fontsize=25)
plt.xlabel('Starting year of Congressional session', fontsize=25)
plt.tight_layout()
plt.savefig(Path(output_path, 'speech_count.pdf'), format='pdf', dpi=300)
plt.savefig(Path(output_path, 'speech_count.png'), format='png', dpi=300)
plt.savefig(Path(output_path, 'speech_count.svg'), format='svg', dpi=300)
