import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rblib import mplconfig


def total_sum_normalize(dt, columns):
    # 计算每个样本列的总和
    sample_sums = dt[columns].sum(axis=0)
    # 计算所有样本总和的均值
    global_mean = sample_sums.mean()
    for col in columns:
        scale_factor = 1 / sample_sums[col] # global_mean/sample_sums[col]
        dt[col] = dt[col] * scale_factor
    return dt


# read gene list
f = open(sys.argv[1],'r') # Fig1B_gene_list.txt
gene_list = []
for line in f:
    gene_list.append(line.rstrip('\n'))
f.close()

# read files
df1 = pd.read_csv('../Raw_data/Hela/Palmitoylayion_Protein_Hela.csv') 
df2 = pd.read_csv('../Raw_data/Hela/Protein_Expression_Hela.csv')
df3 = pd.read_csv('../Raw_data/PANC-1/Palmitoylayion_Protein_PANC1.csv')
df4 = pd.read_csv('../Raw_data/PANC-1/Protein_Expression_PANC1.csv')

## hela
df1_tmp = df1[df1['GeneSymbol'].isin(gene_list)]
df2_tmp = df2[df2['Gene Symbol'].isin(gene_list)]
# 指定要计算均值的列
mean_cols = ['WT1', 'WT2', 'WT3', 'KO1', 'KO2', 'KO3']
df1_tmp = total_sum_normalize(df1_tmp, mean_cols)
df2_tmp = total_sum_normalize(df2_tmp, mean_cols)

# 计算每行的均值
df1_tmp['row_mean'] = df1_tmp[mean_cols].mean(axis=1)
idx_max = df1_tmp.groupby('GeneSymbol')['row_mean'].idxmax()
df1_indexed = df1_tmp.loc[idx_max].drop(columns=['row_mean'])
df1_indexed = df1_indexed.set_index('GeneSymbol')

df2_tmp['row_mean'] = df2_tmp[mean_cols].mean(axis=1)
idx_max = df2_tmp.groupby('Gene Symbol')['row_mean'].idxmax()
df2_indexed= df2_tmp.loc[idx_max].drop(columns=['row_mean'])
df2_indexed = df2_indexed.set_index('Gene Symbol')

wt1 = np.log10(df1_indexed['WT1']+1) - np.log10(df2_indexed['WT1']+1)
wt2 = np.log10(df1_indexed['WT2']+1) - np.log10(df2_indexed['WT2']+1)
wt3 = np.log10(df1_indexed['WT3']+1) - np.log10(df2_indexed['WT3']+1)
ko1 = np.log10(df1_indexed['KO1']+1) - np.log10(df2_indexed['KO1']+1)
ko2 = np.log10(df1_indexed['KO2']+1) - np.log10(df2_indexed['KO2']+1)
ko3 = np.log10(df1_indexed['KO3']+1) - np.log10(df2_indexed['KO3']+1)

df_hela = pd.DataFrame({'Gene':df1_indexed.index,'HeLa WT-1':wt1,'HeLa WT-2':wt2,'HeLa WT-3':wt3,'HeLa KO-1':ko1,'HeLa KO-2':ko2,'HeLa KO-3':ko3})

## panc-1
df3_tmp = df3[df3['GeneSymbol'].isin(gene_list)]
df4_tmp = df4[df4['Gene Symbol'].isin(gene_list)]
# 指定要计算均值的列
mean_cols = ['ctrl1', 'ctrl2', 'ctrl3', 'exp1', 'exp2', 'exp3']
df3_tmp = total_sum_normalize(df3_tmp, mean_cols)
df4_tmp = total_sum_normalize(df4_tmp, mean_cols)

# 计算每行的均值
df3_tmp['row_mean'] = df3_tmp[mean_cols].mean(axis=1)
idx_max = df3_tmp.groupby('GeneSymbol')['row_mean'].idxmax()
df3_indexed= df3_tmp.loc[idx_max].drop(columns=['row_mean'])
df3_indexed = df3_indexed.set_index('GeneSymbol')

df4_tmp['row_mean'] = df4_tmp[mean_cols].mean(axis=1)
idx_max = df4_tmp.groupby('Gene Symbol')['row_mean'].idxmax()
df4_indexed= df4_tmp.loc[idx_max].drop(columns=['row_mean'])
df4_indexed = df4_indexed.set_index('Gene Symbol')

wt1 = np.log10(df3_indexed['ctrl1']+1) - np.log10(df4_indexed['ctrl1']+1)
wt2 = np.log10(df3_indexed['ctrl2']+1) - np.log10(df4_indexed['ctrl2']+1)
wt3 = np.log10(df3_indexed['ctrl3']+1) - np.log10(df4_indexed['ctrl3']+1)
ko1 = np.log10(df3_indexed['exp1']+1) - np.log10(df4_indexed['exp1']+1)
ko2 = np.log10(df3_indexed['exp2']+1) - np.log10(df4_indexed['exp2']+1)
ko3 = np.log10(df3_indexed['exp3']+1) - np.log10(df4_indexed['exp3']+1)

df_panc = pd.DataFrame({'Gene':df3_indexed.index,'PANC-1 WT-1':wt1,'PANC-1 WT-2':wt2,'PANC-1 WT-3':wt3,'PANC-1 OE-1':ko1,'PANC-1 OE-2':ko2,'PANC-1 OE-3':ko3})

# 设置随机种子以便结果可重复
np.random.seed(123)

# 创建DataFrame
df_tmp = pd.merge(df_hela, df_panc, on='Gene', how='inner')
df_tmp = df_tmp.set_index('Gene')
df = df_tmp.dropna()
# heatmap
g = sns.clustermap(df,
                   col_cluster=False,      # 样本不聚类
                   linewidths=0.8,         # 格子边框宽度
                   linecolor='black',      # 边框颜色
                   cmap='vlag',             # 颜色映射，可根据数据调整
                   figsize=(10, 7),         # 整体图形大小
                   dendrogram_ratio=0.15,   # 树状图区域比例
                   cbar_pos=(0.02, 0.85, 0.03, 0.1),  # 颜色条位置 (左, 下, 宽, 高)
                   xticklabels=True,        # 显示样本标签
                   yticklabels=True,        # 显示基因标签
                   )

# 调整样本标签旋转，避免重叠
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')

# 设置坐标轴标签字体大小
g.ax_heatmap.tick_params(axis='both', labelsize=10)

# 显示图形（在Jupyter中会自动显示，脚本中需调用plt.show()）
plt.savefig('output.svg', format='svg', bbox_inches='tight')  # SVG 矢量图
plt.savefig('output.png', format='png', dpi=300, bbox_inches='tight')  # PNG 高分辨率
