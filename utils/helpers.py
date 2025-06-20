import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

def plot_generated(real_data: pd.DataFrame, fake_data: pd.DataFrame, discrete_columns, numerical_columns, log_count: bool = False):
    fig = plt.figure(figsize=(12, 5 * len(numerical_columns) // 2 + 1))
    spec = gridspec.GridSpec(nrows=len(numerical_columns) // 2 + 1, ncols=2, figure=fig, hspace=1, wspace=0.3)

    for i, col in enumerate(numerical_columns):
        # Calculate KDE
        kde1 = gaussian_kde(real_data[col])
        x_range1 = np.linspace(real_data[col].min(), real_data[col].max(), 300)
        density1 = kde1(x_range1)
        
        kde2 = gaussian_kde(fake_data[col])
        x_range2 = np.linspace(fake_data[col].min(), fake_data[col].max(), 300)
        density2 = kde2(x_range2)
        
        # Create subplot
        ax = fig.add_subplot(spec[i // 2, i % 2])
        
        # Plot KDE lines with better styling
        ax.plot(x_range1, density1, linewidth=2, color='steelblue', label='Real', alpha=0.9)
        ax.plot(x_range2, density2, linewidth=2, color='orange', label='Generated', alpha=0.9)
        
        # Fill areas with transparency
        ax.fill_between(x_range1, density1, alpha=0.3, color='steelblue')
        ax.fill_between(x_range2, density2, alpha=0.3, color='orange')
        
        # Styling
        ax.set_title(f'{col.upper()}', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_xlabel('Value', fontsize=11)
        
        # Grid and legend
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right')
        
        # Clean borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Overall title
    fig.suptitle('Komparasi KDE Data Asli vs Data Sintesis berdasarkan kolom', 
                fontsize=16, fontweight='bold', y=0.98)

    # plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(14, 6 * len(discrete_columns)))
    spec = gridspec.GridSpec(nrows=len(discrete_columns), ncols=1, figure=fig, hspace=0.4)

    for i, col in enumerate(discrete_columns):
        # Get data
        cat_counts_real = dict(sorted(real_data[col].value_counts(sort=False).to_dict().items()))
        cat_counts_fake = dict(sorted(fake_data[col].value_counts(sort=False).to_dict().items()))
        
        categories = list(cat_counts_real.keys())
        
        real_vals = np.array(list(cat_counts_real.values()))
        fake_vals = np.array(list(cat_counts_fake.values()))

        if log_count:
            real_vals = np.log(real_vals)
            fake_vals = np.log(fake_vals)
        
        x = np.arange(len(categories))
        width = 0.35
        
        # Create subplot
        ax = fig.add_subplot(spec[i])
        
        # Create bars with better styling
        bars1 = ax.bar(x - width/2, real_vals, width=width, 
                    label="Real", color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, fake_vals, width=width, 
                    label="Generated", color='orange', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        ax.bar_label(bars1, fmt='%.2f' if log_count else '%d', fontsize=8, fontweight='bold')
        ax.bar_label(bars2, fmt='%.2f' if log_count else '%d', fontsize=8, fontweight='bold')
        
        # Styling
        ax.set_title(f'{col.upper()}', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Log Count', fontsize=11)
        ax.set_xlabel('Categories', fontsize=11)
        ax.set_xticks(x, categories)
        
        # Grid and legend
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        if i == 0:
            ax.legend(bbox_to_anchor=(1,1,1,1), loc='lower left')
        
        # Clean borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Rotate x-labels if too many categories
        if len(categories) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Overall title
    fig.suptitle('Komparasi Data Asli vs Data Sintesis by Column', 
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.show()

def prepare_data(df, discrete_columns, label):
    x = pd.get_dummies(df.drop(label, axis=1), columns=discrete_columns, prefix='+').values
    y = df[[label]].values
    return x, y