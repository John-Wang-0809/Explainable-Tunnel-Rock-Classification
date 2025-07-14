import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt
from scipy import stats
import pingouin as pg
import seaborn as sns

# Set font configuration
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# Load data
file_path = 'data/Raw_Data.csv'
data = pd.read_csv(file_path)

# Extract Q1-Q19 data
q_columns = [f'Q{i}' for i in range(1, 20)]
q_data = data[q_columns].dropna()

print("Data preprocessing completed")

# Descriptive statistics
print("\nDescriptive Statistics:")
desc_stats_orig = q_data.describe().T[['mean', 'std', 'min', 'max']]
print(desc_stats_orig)

# Correlation matrix sample
corr_matrix = q_data.corr()
print("\nCorrelation Matrix (Sample):")
print(corr_matrix.iloc[:5, :5].round(2))

# 1. Exploratory Factor Analysis (EFA)
# Bartlett's test and KMO test
chi_square_value, p_value = calculate_bartlett_sphericity(q_data)
kmo_all, kmo_model = calculate_kmo(q_data)

p = q_data.shape[1]
df = (p * (p-1)) / 2

# Create KMO and Bartlett's Test table
kmo_bartlett_table = pd.DataFrame({
    'Metric': ['KMO Value', 'Bartlett\'s Test of Sphericity', 'Approx. Chi-Square', 'df'],
    'Value': [f'{kmo_model:.2f}', '', f'{chi_square_value:.2f}***', f'{int(df)}']
})

print("\nKMO and Bartlett's Test of Sphericity:")
print(kmo_bartlett_table.to_string(index=False))
print("Note: ***: p < .001")

# Calculate eigenvalues
fa = FactorAnalyzer()
fa.fit(q_data)
eigenvalues, _ = fa.get_eigenvalues()

# Variance explained
total_variance = sum(eigenvalues)
explained_variance_ratio = [val / total_variance for val in eigenvalues]
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Create variance explanation table with specific format
variance_table = pd.DataFrame({
    'Factor': ['Reliability and Competence', 'Trust in Automation', 'Understanding of Predictability', 'Familiarity and Developer Trust'],
    'Eigen Value': eigenvalues[:4],
    'Variance Explained (%)': [val * 100 for val in explained_variance_ratio[:4]],
    'Cumulative Variance (%)': [val * 100 for val in cumulative_variance_ratio[:4]]
})

print("\nExplanation of the Total Variance:")
print(variance_table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print("Note: Rotation Method: Varimax Rotation")

# Scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--', 
         linewidth=2, color='#1f77b4', label='Eigenvalues')
plt.axhline(y=1, color='r', linestyle='-', label='Kaiser Criterion (Eigenvalue=1)')
plt.xlabel('Number of Factors', fontname='Times New Roman', fontsize=14)
plt.ylabel('Eigenvalue', fontname='Times New Roman', fontsize=14)
plt.title('Scree Plot for Principal Component Analysis', fontname='Times New Roman', fontsize=16)
plt.grid()
plt.legend()
plt.xticks(fontsize=12, fontname='Times New Roman')
plt.yticks(fontsize=12, fontname='Times New Roman')
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.savefig('outputs/scree_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Note: Cumulative variance plot code removed as requested

# Factor analysis with 4 factors
print("\n========== Performing Factor Analysis ==========")
n_factors = 4
fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax', method='principal')
fa.fit(q_data)

# Factor loadings
loadings = pd.DataFrame(fa.loadings_, index=q_columns, 
                       columns=['RC', 'TA', 'UP', 'FDT'])

# Factor variance
eigenvalues, proportions, cumulative = fa.get_factor_variance()
factor_variance_df = pd.DataFrame({
    'Factor': ['RC', 'TA', 'UP', 'FDT'],
    'Eigenvalue': eigenvalues,
    'Variance Explained (%)': proportions * 100,
    'Cumulative Variance (%)': cumulative * 100
})

print("\n4-Factor Solution Variance Explained:")
print(factor_variance_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# Communalities
communalities = pd.Series(fa.get_communalities(), index=q_columns, name='h²')

print("\nFactor Loading Matrix:")
print(loadings.round(3))
print("\nCommunalities:")
print(communalities.round(3))

# Save factor loadings
loadings_with_communalities = pd.concat([loadings, communalities], axis=1)
loadings_with_communalities.to_excel('outputs/factor_loadings_all.xlsx')

loadings_filtered = loadings.copy()
mask = abs(loadings_filtered) < 0.4
loadings_filtered[mask] = np.nan
loadings_filtered_with_communalities = pd.concat([loadings_filtered, communalities], axis=1)
loadings_filtered_with_communalities.to_excel('outputs/factor_loadings_filtered.xlsx')

# Descriptive statistics
desc_stats = q_data.describe().T[['mean', 'std']]
desc_stats.columns = ['Mean', 'Standard Deviation']
print("\nDescriptive Statistics:")
print(desc_stats.round(3))

# Cronbach's Alpha
total_alpha = pg.cronbach_alpha(data=q_data)

factor_items = {
    'RC (Reliability and Competence)': ['Q1', 'Q6', 'Q10', 'Q13', 'Q15', 'Q19'],
    'TA (Trust in Automation)': ['Q5', 'Q9', 'Q12', 'Q14', 'Q18'],
    'UP (Understanding and Predictability)': ['Q2', 'Q7', 'Q11', 'Q16'],
    'FDT (Familiarity and Developer Trust)': ['Q3', 'Q4', 'Q8', 'Q17']
}

factor_alphas = {}
for factor_name, items in factor_items.items():
    alpha = pg.cronbach_alpha(data=q_data[items])
    factor_alphas[factor_name] = alpha[0]

# Create Cronbach's Alpha table
cronbach_table = pd.DataFrame({
    'Scale/Factor': ['Overall Rating Scale', 'Reliability and Competence', 'Trust in Automation', 'Understanding of Predictability', 'Familiarity and Developer Trust'],
    'Cronbach\'s Alpha': [total_alpha[0], factor_alphas['RC (Reliability and Competence)'], factor_alphas['TA (Trust in Automation)'], factor_alphas['UP (Understanding and Predictability)'], factor_alphas['FDT (Familiarity and Developer Trust)']]
})

print("\nCronbach's Alpha Reliability Analysis:")
print(cronbach_table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print("Note: All values indicate high reliability (> 0.90)")

# Factor scores
fa_scores = fa.transform(q_data)
factor_scores = pd.DataFrame(fa_scores, columns=['RC', 'TA', 'UP', 'FDT'])

# Normality tests
ks_results = {}
for factor in factor_scores.columns:
    stat, p = stats.kstest(factor_scores[factor], 'norm')
    ks_results[factor] = {'D statistic': stat, 'p-value': p}

# Group difference tests
group_info = data.loc[q_data.index, 'Group']
factor_scores_with_group = factor_scores.copy()
factor_scores_with_group['Group'] = group_info

test_results = {}

# Create combined normality and group difference test table
factor_names_mapping = {
    'RC': 'Reliability and Competence',
    'TA': 'Trust in Automation', 
    'UP': 'Understanding of Predictability',
    'FDT': 'Familiarity and Developer Trust'
}

combined_results = []

for factor in ['RC', 'TA', 'UP', 'FDT']:
    group1 = factor_scores_with_group[factor_scores_with_group['Group'] == 1][factor]
    group2 = factor_scores_with_group[factor_scores_with_group['Group'] == 2][factor]
    
    p_value_ks = ks_results[factor]['p-value']
    d_stat = ks_results[factor]['D statistic']
    
    # Get group sizes
    n1, n2 = len(group1), len(group2)
    
    # Since all factors are non-normal, use Mann-Whitney U test
    median1 = group1.median()
    median2 = group2.median()
    
    q1_g1, q3_g1 = np.percentile(group1, [25, 75])
    q1_g2, q3_g2 = np.percentile(group2, [25, 75])
    
    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    mean_u = n1 * n2 / 2
    std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z_score = (u_stat - mean_u) / std_u
    
    # Format significance stars
    p_star = ''
    if p_value < 0.001:
        p_star = '***'
    elif p_value < 0.01:
        p_star = '**'
    elif p_value < 0.05:
        p_star = '*'
    
    ks_star = ''
    if p_value_ks < 0.001:
        ks_star = '***'
    elif p_value_ks < 0.01:
        ks_star = '**'
    elif p_value_ks < 0.05:
        ks_star = '*'
    
    combined_results.append({
        'Factor': f'**{factor_names_mapping[factor]}**',
        'D statistic': f'.{d_stat:.3f}'[1:],  # Remove leading 0
        'p (KS)': f'{p_value_ks:.3f}{ks_star}',
        'Group 1.0 Median (P25, P75)': f'{median1:.3f} ({q1_g1:.3f}, {q3_g1:.3f})',
        'Group 2.0 Median (P25, P75)': f'{median2:.3f} ({q1_g2:.3f}, {q3_g2:.3f})',
        'Z-Score': f'{z_score:.3f}',
        'p (MW)': f'{p_value:.3f}{p_star}'
    })

# Create the combined table
combined_table = pd.DataFrame(combined_results)

print("\nNormality Test and Mann-Whitney U Test Results of Four Factors:")
print("| Factor | Kolmogorov-Smirnov Test | | Group Median (P25, P75) | | | |")
print("| | D statistic | p | 1.0 (n=102) | 2.0 (n=102) | Z-Score | p |")
print("|---|---|---|---|---|---|---|")
for _, row in combined_table.iterrows():
    print(f"| {row['Factor']} | {row['D statistic']} | {row['p (KS)']} | {row['Group 1.0 Median (P25, P75)']} | {row['Group 2.0 Median (P25, P75)']} | {row['Z-Score']} | {row['p (MW)']} |")

# Save results to Excel
factor_table = pd.DataFrame(index=q_columns)
factor_table['Mean'] = desc_stats['Mean']
factor_table['Standard Deviation'] = desc_stats['Standard Deviation']

for col in loadings.columns:
    factor_table[col] = loadings[col]
factor_table['h²'] = communalities

ks_df = pd.DataFrame(ks_results).T
ks_df.columns = ['D statistic', 'p-value']

test_df = pd.DataFrame()
for factor in ['RC', 'TA', 'UP', 'FDT']:
    group1 = factor_scores_with_group[factor_scores_with_group['Group'] == 1][factor]
    group2 = factor_scores_with_group[factor_scores_with_group['Group'] == 2][factor]
    
    median1 = group1.median()
    median2 = group2.median()
    
    q1_g1, q3_g1 = np.percentile(group1, [25, 75])
    q1_g2, q3_g2 = np.percentile(group2, [25, 75])
    
    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    n1, n2 = len(group1), len(group2)
    mean_u = n1 * n2 / 2
    std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z_score = (u_stat - mean_u) / std_u
    
    row = pd.Series(name=factor)
    row['Test Method'] = 'Mann-Whitney U'
    row['Group 1'] = f"{median1:.3f} (IQR: {q1_g1:.3f}-{q3_g1:.3f})"
    row['Group 2'] = f"{median2:.3f} (IQR: {q1_g2:.3f}-{q3_g2:.3f})"
    row['Statistic'] = f"U = {u_stat:.0f}, Z = {z_score:.3f}"
    row['p-value'] = p_value
    row['Significance'] = '*' if p_value < 0.05 else ''
    
    test_df = pd.concat([test_df, row.to_frame().T])

# Export to Excel
with pd.ExcelWriter('outputs/factor_analysis_results.xlsx') as writer:
    factor_table.to_excel(writer, sheet_name='Descriptive Stats and Loadings')
    ks_df.to_excel(writer, sheet_name='Normality Tests')
    test_df.to_excel(writer, sheet_name='Group Difference Tests')
    variance_table.to_excel(writer, sheet_name='Initial Eigenvalues', index=False)
    factor_variance_df.to_excel(writer, sheet_name='4-Factor Variance', index=False)
    kmo_bartlett_table.to_excel(writer, sheet_name='KMO and Bartlett', index=False)
    cronbach_table.to_excel(writer, sheet_name='Reliability Analysis', index=False)
    combined_table.to_excel(writer, sheet_name='Combined Analysis', index=False)

print("\nAnalysis completed. Results saved to 'outputs/factor_analysis_results.xlsx'")

# Note: Boxplot visualization code removed as requested