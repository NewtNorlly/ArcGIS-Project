import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Set global academic style
plt.rcParams['font.family'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelpad'] = 10
sns.set_context("paper", font_scale=1.2)
plt.style.use('seaborn-v0_8-white')

# 1. Advanced Factor Importance (Academic Style)
factor_df = pd.read_csv('探测因子报表.xlsx - Factor detector.csv')
factor_df = factor_df.sort_values('q', ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
colors = sns.color_palette("Blues_d", n_colors=len(factor_df))
ax.barh(factor_df['Variable'], factor_df['q'], color=colors, height=0.6, edgecolor='black', linewidth=0.8)

# Add p-value stars (all seem < 0.01 or ~0.08)
for i, (q, p) in enumerate(zip(factor_df['q'], factor_df['p-value'])):
    star = ""
    try:
        p_val = float(p) if '<' not in str(p) else 0.0
    except:
        p_val = 1.0
    if p_val < 0.01 or '<0.01' in str(p): star = "***"
    elif p_val < 0.05: star = "**"
    elif p_val < 0.1: star = "*"
    ax.text(q + 0.002, i, f"{q:.3f}{star}", va='center', fontweight='bold')

ax.set_title('建设用地扩张驱动因子探测结果 ($q$值)', fontsize=16, pad=20)
ax.set_xlabel('解释力 ($q$)', fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('academic_factor_detector.png', dpi=300)
plt.close()

# 2. Advanced Interaction Detector Matrix
# Load Interaction Data
inter_raw = pd.read_csv('探测因子报表.xlsx - Interaction detector.csv')
headers = inter_raw.iloc[0, 1:6].values.tolist()
matrix_data = inter_raw.iloc[1:6, 1:6].values.astype(float)
inter_matrix = pd.DataFrame(matrix_data, index=headers, columns=headers)

# Get individual q values for comparison
q_vals = factor_df.set_index('Variable')['q'].to_dict()

# Calculate Interaction Types
# types: 0: Independent, 1: Weaken, 2: Uni-weaken, 3: Bi-enhance, 4: Nonlinear-enhance
type_matrix = np.zeros_like(matrix_data, dtype=object)
for i in range(len(headers)):
    for j in range(len(headers)):
        if i > j:
            q12 = matrix_data[i, j]
            q1 = q_vals.get(headers[i], 0)
            q2 = q_vals.get(headers[j], 0)
            if q12 > (q1 + q2):
                type_matrix[i, j] = 'NE' # Nonlinear Enhance
            elif q12 > max(q1, q2):
                type_matrix[i, j] = 'BE' # Bi-variable Enhance
            else:
                type_matrix[i, j] = 'E'

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(inter_matrix, dtype=bool))
sns.heatmap(inter_matrix, annot=True, mask=mask, cmap='YlGnBu', fmt=".3f", 
            linewidths=1, linecolor='white', cbar_kws={'label': '交互$q$值'},
            annot_kws={"size": 12, "weight": "bold"})

# Add Type Labels (NE/BE) in the cells
for i in range(len(headers)):
    for j in range(len(headers)):
        if i > j:
            label = type_matrix[i, j]
            ax.text(j + 0.5, i + 0.8, f"({label})", ha='center', va='center', fontsize=10, color='darkred')

ax.set_title('驱动因子交互作用探测矩阵 (NE: 非线性增强, BE: 双因子增强)', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('advanced_interaction_matrix.png', dpi=300)
plt.close()

# 3. Land Use Transfer Matrix (Polished Heatmap)
land_use_raw = pd.read_csv('土地利用报表.xlsx - Sheet1.csv')
transfer_data = land_use_raw.iloc[13:22, 2:11].values.astype(float)
categories = land_use_raw.iloc[12, 2:11].values.tolist()
transfer_df = pd.DataFrame(transfer_data, index=categories, columns=categories)

# Advanced Heatmap with log scale color for better visibility of small transfers
from matplotlib.colors import LogNorm
plt.figure(figsize=(12, 10))
# Mask zeros or very small values for cleaner look
annot_data = transfer_df.applymap(lambda x: f"{x:.1f}" if x > 10 else "")

sns.heatmap(transfer_df, annot=annot_data.values, fmt="", cmap="Reds", 
            linewidths=0.5, linecolor='#eeeeee', cbar_kws={'label': '转移面积 (km²)'},
            norm=LogNorm(vmin=1, vmax=transfer_df.values.max()))

plt.title('2000-2020 长三角土地利用空间转移矩阵 (km²)', fontsize=18, pad=25)
plt.xlabel('2020年地类', fontsize=14, labelpad=15)
plt.ylabel('2000年地类', fontsize=14, labelpad=15)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('academic_transfer_matrix.png', dpi=300)
plt.close()

# 4. Construction Land Growth (Advanced Distribution - Raincloud Plot style)
construction_df = pd.read_csv('建设用地情况统计.xlsx - 县界.csv')
# Keep only Top 15 cities or aggregate
top_cities = construction_df.groupby('市')['建设用地变化率'].median().sort_values(ascending=False).index

plt.figure(figsize=(15, 8))
# Strip plot + Box plot
sns.stripplot(data=construction_df, x='市', y='建设用地变化率', order=top_cities, 
              size=3, color="gray", alpha=0.4, jitter=0.2)
sns.boxplot(data=construction_df, x='市', y='建设用地变化率', order=top_cities, 
            whis=[0, 100], width=.6, palette="vlag", boxprops=dict(alpha=.7))

plt.title('长三角各市县域建设用地变化率分布特征', fontsize=18, pad=20)
plt.ylabel('建设用地变化率 (%)', fontsize=14)
plt.xlabel('城市', fontsize=14)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('construction_growth_distribution.png', dpi=300)
plt.close()


