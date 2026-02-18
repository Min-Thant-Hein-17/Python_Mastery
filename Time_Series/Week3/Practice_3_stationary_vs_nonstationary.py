# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 12:06:14 2026

@author: ASUS
"""


import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("E:\Time_Series\Time_Series_Coding\Week3\stationary_vs_nonstationary_data.csv")
df['date'] = pd.to_datetime(df['date'])

mean_stat = df['stationary'].mean()

df['non_stat_rolling'] = df['non_stationary'].rolling(window=30).mean()

# 2. Create the figure and subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# --- Top Plot: Stationary Data ---
axes[0].plot(df['date'], df['stationary'], color='tab:blue', linewidth=1, label='Stationary Series')
# Add the red horizontal line for the mean
axes[0].axhline(y=mean_stat, color='red', linestyle='--', linewidth=2, label=f'Mean ({mean_stat:.1f})')
axes[0].set_title('Stationary Time Series', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Value')
axes[0].grid(True, which='both', linestyle=':', alpha=0.7) # Add grid
axes[0].legend(loc='upper right')


# --- Plot 2: Non-Stationary Data ---
axes[1].plot(df['date'], df['non_stationary'], color='orange', alpha=0.5, label='Actual Data', linewidth=1)
# Red line: Rolling mean follows the trend for non-stationary data
axes[1].plot(df['date'], df['non_stat_rolling'], color='red', linewidth=2, label='Rolling Mean (Variable)')
axes[1].set_title('Non-Stationary Time Series', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Value')
axes[1].set_xlabel('Date')
axes[1].grid(True, which='both', linestyle=':', alpha=0.7)
axes[1].legend(loc='upper right')

# 3. Fine-tuning layout
plt.tight_layout()
plt.show()
