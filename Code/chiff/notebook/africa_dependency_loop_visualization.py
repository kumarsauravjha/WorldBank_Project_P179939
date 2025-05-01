import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 150

# Create dependency loop visualization
plt.figure(figsize=(14, 8))

# Define the 3 stages of the loop
stages = ['African Exports\n(Raw Materials)', 'Global Processing\n(Value Addition)', 'African Imports\n(Processed Goods)']
positions = [1, 2, 3]

# First create the main flow blocks
plt.bar(positions, [0.7, 0.9, 0.7], bottom=[0.2, 0.1, 0.2], width=0.6, color=['lightblue', 'lightgreen', 'salmon'], edgecolor='black', linewidth=1.5)

# Add annotations for each stage
plt.annotate('High Freight Rates', xy=(1, 0.6), xytext=(1, 0.75), ha='center', fontsize=12, 
             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

plt.annotate('Captured Value', xy=(2, 0.6), xytext=(2, 0.75), ha='center', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

plt.annotate('Marked-up Prices', xy=(3, 0.6), xytext=(3, 0.75), ha='center', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

# Add examples
plt.text(1, 0.35, 'Example: Cocoa, Crude Oil,\nMinerals', ha='center', fontsize=10)
plt.text(2, 0.25, 'Example: Chocolate, Gasoline,\nCanned Food Production', ha='center', fontsize=10)
plt.text(3, 0.35, 'Example: Imported Processed Foods,\nRefined Fuels', ha='center', fontsize=10)

# Add arrows to show the cycle
plt.annotate('', xy=(1.4, 0.5), xytext=(0.6, 0.5), arrowprops=dict(arrowstyle='->', color='blue', lw=2))
plt.annotate('', xy=(1.6, 0.5), xytext=(2.4, 0.5), arrowprops=dict(arrowstyle='->', color='blue', lw=2))
plt.annotate('', xy=(2.6, 0.5), xytext=(3.4, 0.5), arrowprops=dict(arrowstyle='->', color='blue', lw=2))

# Closing the loop with a return arrow
plt.annotate('', xy=(0.6, 0.3), xytext=(3.4, 0.3), arrowprops=dict(arrowstyle='->', color='red', lw=2, ls='--'))

# Add the key insights from the data
plt.text(0.7, 0.9, 'KEY INSIGHTS FROM DATA:', fontsize=12, fontweight='bold')
plt.text(2, 0.85, '• Africa exports primarily raw materials (crude oil, minerals, crops)', fontsize=10, ha='center')
plt.text(2, 0.82, '• High shipping costs make processed exports uneconomical', fontsize=10, ha='center')
plt.text(2, 0.79, '• Top destinations: China, India, USA add value through processing', fontsize=10, ha='center')
plt.text(2, 0.76, '• Africa then imports processed versions at higher costs', fontsize=10, ha='center')

# Add consequences below the loop
plt.text(0.7, 0.15, 'CONSEQUENCES:', fontsize=12, fontweight='bold')
plt.text(2, 0.1, '• Reinforces commodity dependence\n• Limits local industrial development\n• Loss of potential jobs and revenue\n• Higher consumer prices', 
         fontsize=11, ha='center')

# Set plot details
plt.title('Africa\'s Trade Dependency Loop', fontsize=18, pad=20)
plt.ylim(0, 1)
plt.xlim(0, 4)
plt.xticks(positions, stages, fontsize=12)
plt.yticks([])
plt.box(False)
plt.tight_layout()

# Save the figure
plt.savefig('dependency_loop.png', dpi=150, bbox_inches='tight')
print('Dependency loop visualization saved as dependency_loop.png') 