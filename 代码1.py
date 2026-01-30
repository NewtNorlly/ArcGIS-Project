import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl


# 设置中文字体
plt.rcParams['font.family'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 因子探测器 (Factor Detector) - 高级学术风格 Lollipop Chart
factor_df = pd.read_csv('探测因子报表.xlsx - Factor detector.csv')
factor_df = factor_df.sort_values('q', ascending=True)

plt.figure(figsize=(10, 6))
# 使用更学术的色调
my_color = "#34495e"
plt.hlines(y=factor_df['Variable'], xmin=0, xmax=factor_df['q'], color='lightgrey', linewidth=1)
plt.scatter(factor_df['q'], factor_df['Variable'], s=150, color="#e74c3c", edgecolors=my_color, zorder=3)

# 在点旁边标注 q 值
for i, q in enumerate(factor_df['q']):
    plt.text(q + 0.005, i, f'{q:.4f}', va='center', fontsize=11, fontweight='bold', color=my_color)

plt.title('建设用地扩张驱动因子影响力排名 ($q$ 值)', fontsize=16, pad=20)
plt.xlabel('解释力 ($q$)', fontsize=12)
plt.ylabel('驱动因子', fontsize=12)
plt.xlim(0, factor_df['q'].max() * 1.2)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('advanced_factor_importance.png', dpi=300)
plt.close()

# 2. 交互探测器 (Interaction Detector) - 高级学术热力图
# 读取交互数据
inter_raw = pd.read_csv('探测因子报表.xlsx - Interaction detector.csv')
cols = inter_raw.iloc[0, 1:6].values.tolist()
matrix_data = inter_raw.iloc[1:6, 1:6].values.astype(float)
inter_matrix = pd.DataFrame(matrix_data, index=cols, columns=cols)

# 获取单因子 q 值用于判断交互类型
q_dict = dict(zip(factor_df['Variable'], factor_df['q']))

# 构造交互类型标注
# 定义逻辑：Nonlinear-enhance (NE), Bi-enhance (BE)
annot_matrix = np.full(inter_matrix.shape, "", dtype=object)
for i in range(len(cols)):
    for j in range(i):
        q1 = q_dict[cols[i]]
        q2 = q_dict[cols[j]]
        q_inter = inter_matrix.iloc[i, j]
        if q_inter > (q1 + q2):
            type_str = f"{q_inter:.3f}\n(NE)" # Nonlinear Enhance
        elif q_inter > max(q1, q2):
            type_str = f"{q_inter:.3f}\n(BE)" # Bi-enhance
        else:
            type_str = f"{q_inter:.3f}"
        annot_matrix[i, j] = type_str

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(inter_matrix, dtype=bool))
sns.heatmap(inter_matrix, mask=mask, annot=annot_matrix, fmt="", cmap="YlGnBu", 
            square=True, cbar_kws={"shrink": .8, "label": "交互 $q$ 值"}, 
            linewidths=1, linecolor='white', annot_kws={"size": 10})

plt.title('因子交互作用强度及类型 (NE: 非线性增强, BE: 双因子增强)', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('advanced_interaction_heatmap.png', dpi=300)
plt.close()

# 3. 土地利用转移矩阵 (Land Use Transfer) - 细致热力图
land_use_raw = pd.read_csv('土地利用报表.xlsx - Sheet1.csv')
transfer_data = land_use_raw.iloc[13:22, 2:11].values.astype(float)
categories = land_use_raw.iloc[12, 2:11].values.tolist()
transfer_df = pd.DataFrame(transfer_data, index=categories, columns=categories)

# 计算百分比转移矩阵
transfer_percent = transfer_df.div(transfer_df.sum(axis=1), axis=0) * 100

plt.figure(figsize=(12, 10))
sns.heatmap(transfer_df, annot=True, fmt=".1f", cmap="Reds", cbar_kws={'label': '转移面积 (km²)'})
# 覆盖一层数值，如果是重点转移（比如到建设用地），可以特别标注
plt.title('2000-2020 长三角土地利用空间转移矩阵 (km²)', fontsize=16, pad=20)
plt.xlabel('2020年地类', fontsize=12)
plt.ylabel('2000年地类', fontsize=12)
plt.tight_layout()
plt.savefig('advanced_transfer_matrix.png', dpi=300)
plt.close()

# 4. 城市建设用地变化率 - 雨靶图 (Raincloud-like plot: Box + Strip)
construction_df = pd.read_csv('建设用地情况统计.xlsx - 县界.csv')
city_order = construction_df.groupby('市')['建设用地变化率'].median().sort_values(ascending=False).index

plt.figure(figsize=(16, 8))
# 绘制箱线图
sns.boxplot(data=construction_df, x='市', y='建设用地变化率', order=city_order, 
            whis=[0, 100], width=.6, palette="vlag", linewidth=1)
# 叠加散点（抖动）
sns.stripplot(data=construction_df, x='市', y='建设用地变化率', order=city_order, 
              size=3, color=".3", linewidth=0, alpha=0.4)

plt.title('长三角各市县域建设用地变化率分布 (学术风格)', fontsize=16, pad=20)
plt.ylabel('建设用地变化率 (%)', fontsize=12)
plt.xlabel('城市', fontsize=12)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('advanced_construction_variation.png', dpi=300)
plt.close()


