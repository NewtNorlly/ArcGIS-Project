import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm


# Font Setup
font_path = '/usr/share/fonts/NotoSansCJK-Regular.ttc'
prop = fm.FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False

# Helper for labels
def set_label_font(ax, title=None, xlabel=None, ylabel=None):
    if title: ax.set_title(title, fontproperties=prop, fontsize=16, pad=15)
    if xlabel: ax.set_xlabel(xlabel, fontproperties=prop, fontsize=13)
    if ylabel: ax.set_ylabel(ylabel, fontproperties=prop, fontsize=13)
    for label in ax.get_xticklabels(): label.set_fontproperties(prop)
    for label in ax.get_yticklabels(): label.set_fontproperties(prop)

# 1. Academic Factor Importance
factor_df = pd.read_csv('探测因子报表.xlsx - Factor detector.csv').sort_values('q', ascending=True)
fig, ax = plt.subplots(figsize=(9, 6))
colors = sns.color_palette("mako", n_colors=len(factor_df))
bars = ax.barh(factor_df['Variable'], factor_df['q'], color=colors, edgecolor='0.3', alpha=0.85)

for i, (q, p) in enumerate(zip(factor_df['q'], factor_df['p-value'])):
    sig = "***" if ('<0.01' in str(p) or (isinstance(p, (int, float)) and p < 0.01)) else ("**" if (isinstance(p, (int, float)) and p < 0.05) else "")
    ax.text(q + 0.005, i, f"{q:.3f}{sig}", va='center', fontsize=11, fontweight='bold')

set_label_font(ax, "建设用地扩张主导因子解释力 ($q$值)", "解释力 ($q$)", "驱动因子")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('academic_factor_q.png', dpi=300)
plt.close()

# 2. Advanced Interaction Matrix (Heatmap)
inter_raw = pd.read_csv('探测因子报表.xlsx - Interaction detector.csv')
headers = inter_raw.iloc[0, 1:6].values.tolist()
matrix_data = inter_raw.iloc[1:6, 1:6].values.astype(float)
q_dict = factor_df.set_index('Variable')['q'].to_dict()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(matrix_data, dtype=bool))
sns.heatmap(matrix_data, mask=mask, annot=True, fmt=".3f", cmap="YlGnBu", 
            xticklabels=headers, yticklabels=headers,
            cbar_kws={'label': '交互作用解释力 (q)'},
            annot_kws={"size": 11, "weight": "bold"})

# Annotate Interaction Types
for i in range(len(headers)):
    for j in range(len(headers)):
        if i > j:
            q12 = matrix_data[i, j]
            q1, q2 = q_dict[headers[i]], q_dict[headers[j]]
            label = "NE" if q12 > (q1 + q2) else "BE"
            ax.text(j + 0.5, i + 0.8, f"({label})", ha='center', va='center', fontsize=9, color='darkred', alpha=0.8)

set_label_font(ax, "驱动因子交互作用增强矩阵", "因子 X", "因子 Y")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('advanced_interaction_heatmap.png', dpi=300)
plt.close()

# 3. Refined Transfer Matrix
land_use_raw = pd.read_csv('土地利用报表.xlsx - Sheet1.csv')
trans_data = land_use_raw.iloc[13:22, 2:11].values.astype(float)
cats = land_use_raw.iloc[12, 2:11].values.tolist()

plt.figure(figsize=(11, 9))
# We use a threshold to simplify the display
display_data = np.where(trans_data > 1.0, trans_data, 0)
sns.heatmap(display_data, annot=True, fmt=".1f", cmap="OrRd", 
            xticklabels=cats, yticklabels=cats, mask=(display_data == 0),
            linewidths=1, linecolor='whitesmoke')
ax = plt.gca()
set_label_font(ax, "2000-2020 土地利用类型转换矩阵 (km²)", "2020年地类", "2000年地类")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('advanced_transfer_matrix.png', dpi=300)
plt.close()

# 4. Multi-level City Growth Distribution
const_df = pd.read_csv('建设用地情况统计.xlsx - 县界.csv')
top_cities = const_df.groupby('市')['建设用地变化率'].median().sort_values(ascending=False).head(20).index
subset = const_df[const_df['市'].isin(top_cities)]

plt.figure(figsize=(14, 7))
sns.violinplot(data=subset, x='市', y='建设用地变化率', order=top_cities, 
               palette="husl", inner="quart", cut=0, alpha=0.6)
sns.stripplot(data=subset, x='市', y='建设用地变化率', order=top_cities, 
              color="black", size=3, alpha=0.3, jitter=True)
ax = plt.gca()
set_label_font(ax, "建设用地变化率最高的20个城市分布情况", "城市", "建设用地变化率 (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('city_growth_distribution.png', dpi=300)
plt.close()


