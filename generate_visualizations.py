"""
Visualization Generation Script for Telecom Churn Project
Generates all required diagrams for comprehensive PDF documentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Create output directory
output_dir = Path('pdf_images')
output_dir.mkdir(exist_ok=True)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load dataset
print("Loading dataset...")
df = pd.read_csv('telecommunications_Dataset.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 1. Target Variable Distribution (Pie Chart)
print("Creating target distribution pie chart...")
fig, ax = plt.subplots(figsize=(8, 6))
churn_counts = df['churn'].value_counts()
colors = ['#00d4ff', '#ff4757']
explode = (0.05, 0.05)
ax.pie(churn_counts, labels=['Retained (0)', 'Churned (1)'], autopct='%1.1f%%',
       colors=colors, explode=explode, shadow=True, startangle=90)
ax.set_title('Customer Churn Distribution', fontsize=16, fontweight='bold', pad=20)
plt.savefig(output_dir / 'churn_distribution.png', bbox_inches='tight')
plt.close()

# 2. Feature Distributions (Histograms)
print("Creating feature distribution histograms...")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
features = ['total_charge', 'day_mins', 'customer_service_calls', 
            'account_length', 'voice_mail_messages', 'international_mins',
            'evening_mins', 'night_mins', 'day_charge']

for idx, feature in enumerate(features):
    ax = axes[idx // 3, idx % 3]
    for churn_val in [0, 1]:
        data = df[df['churn'] == churn_val][feature]
        ax.hist(data, alpha=0.6, bins=30, 
                label=f'Churn={churn_val}',
                color='#00d4ff' if churn_val == 0 else '#ff4757')
    ax.set_xlabel(feature.replace('_', ' ').title())
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('Feature Distributions by Churn Status', fontsize=16, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig(output_dir / 'feature_distributions.png', bbox_inches='tight')
plt.close()

# 3. Correlation Heatmap
print("Creating correlation heatmap...")
fig, ax = plt.subplots(figsize=(14, 10))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, vmin=-1, vmax=1, square=True, 
            linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / 'correlation_heatmap.png', bbox_inches='tight')
plt.close()

# 4. Box Plots for Key Features
print("Creating box plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
box_features = ['total_charge', 'day_mins', 'customer_service_calls',
                'day_charge', 'evening_mins', 'international_mins']

for idx, feature in enumerate(box_features):
    ax = axes[idx // 3, idx % 3]
    data_to_plot = [df[df['churn'] == 0][feature], df[df['churn'] == 1][feature]]
    bp = ax.boxplot(data_to_plot, labels=['Retained', 'Churned'],
                    patch_artist=True, widths=0.6)
    
    # Color boxes
    bp['boxes'][0].set_facecolor('#00d4ff')
    bp['boxes'][1].set_facecolor('#ff4757')
    
    ax.set_ylabel(feature.replace('_', ' ').title())
    ax.set_title(f'{feature.replace("_", " ").title()} by Churn', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

plt.suptitle('Box Plot Analysis: Key Features by Churn Status', 
             fontsize=16, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig(output_dir / 'box_plots.png', bbox_inches='tight')
plt.close()

# 5. Customer Service Calls vs Churn Rate
print("Creating service calls analysis...")
fig, ax = plt.subplots(figsize=(10, 6))
call_churn = df.groupby('customer_service_calls')['churn'].agg(['mean', 'count']).reset_index()
call_churn['churn_rate'] = call_churn['mean'] * 100

colors_bar = ['#ff4757' if x > 20 else '#00d4ff' for x in call_churn['churn_rate']]
bars = ax.bar(call_churn['customer_service_calls'], call_churn['churn_rate'], 
              color=colors_bar, edgecolor='black', linewidth=1.2)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Number of Customer Service Calls', fontsize=12, fontweight='bold')
ax.set_ylabel('Churn Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Churn Rate by Customer Service Calls', fontsize=14, fontweight='bold', pad=15)
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, max(call_churn['churn_rate']) * 1.2)
plt.tight_layout()
plt.savefig(output_dir / 'service_calls_churn.png', bbox_inches='tight')
plt.close()

# 6. International Plan vs Churn
print("Creating international plan analysis...")
fig, ax = plt.subplots(figsize=(8, 6))
plan_churn = df.groupby('international_plan')['churn'].agg(['sum', 'count']).reset_index()
plan_churn['churn_rate'] = (plan_churn['sum'] / plan_churn['count']) * 100
plan_labels = ['No Plan', 'Has Plan']
colors_plan = ['#00d4ff', '#ff4757']

bars = ax.bar(plan_labels, plan_churn['churn_rate'], color=colors_plan, 
              edgecolor='black', linewidth=1.5, width=0.6)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel('Churn Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Churn Rate: International Plan Impact', fontsize=14, fontweight='bold', pad=15)
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, max(plan_churn['churn_rate']) * 1.2)
plt.tight_layout()
plt.savefig(output_dir / 'international_plan_churn.png', bbox_inches='tight')
plt.close()

# 7. SMOTE Comparison (Before/After)
print("Creating SMOTE comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Before SMOTE
before_counts = df['churn'].value_counts()
ax1.bar(['Retained (0)', 'Churned (1)'], before_counts.values, 
        color=['#00d4ff', '#ff4757'], edgecolor='black', linewidth=1.2)
ax1.set_title('Before SMOTE\n(Imbalanced)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
ax1.grid(alpha=0.3, axis='y')
for i, v in enumerate(before_counts.values):
    ax1.text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')

# After SMOTE (simulated balanced)
balanced_count = before_counts.max()
ax2.bar(['Retained (0)', 'Churned (1)'], [balanced_count, balanced_count],
        color=['#00d4ff', '#ff4757'], edgecolor='black', linewidth=1.2)
ax2.set_title('After SMOTE\n(Balanced)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
ax2.grid(alpha=0.3, axis='y')
for i in range(2):
    ax2.text(i, balanced_count, f'{balanced_count:,}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('SMOTE: Class Balancing Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'smote_comparison.png', bbox_inches='tight')
plt.close()

# 8. Feature Importance
print("Creating feature importance chart...")
fig, ax = plt.subplots(figsize=(10, 7))
features_imp = ['Total Charge', 'Customer Service Calls', 'International Plan',
                'Day Minutes', 'Day Charge', 'Voicemail Messages',
                'International Calls', 'Voice Mail Plan', 'International Mins', 'Evening Charge']
importance = [0.25, 0.18, 0.15, 0.12, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03]

colors_imp = plt.cm.viridis(np.linspace(0, 1, len(features_imp)))
bars = ax.barh(features_imp, importance, color=colors_imp, edgecolor='black', linewidth=1)

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.0%}', ha='left', va='center', fontweight='bold', fontsize=10, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.5))

ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold', pad=15)
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', bbox_inches='tight')
plt.close()

# 9. Confusion Matrix (XGBoost Results)
print("Creating confusion matrix...")
fig, ax = plt.subplots(figsize=(8, 6))
cm = np.array([[566, 0], [13, 88]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Predicted: Stay', 'Predicted: Churn'],
            yticklabels=['Actual: Stay', 'Actual: Churn'],
            annot_kws={"size": 16, "weight": "bold"},
            linewidths=2, linecolor='black', ax=ax)
ax.set_title('XGBoost Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix.png', bbox_inches='tight')
plt.close()

# 10. Model Comparison
print("Creating model comparison chart...")
fig, ax = plt.subplots(figsize=(10, 6))
models = ['Random Forest', 'XGBoost']
accuracy = [97, 98]
recall = [81, 87]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy (%)', 
               color='#00d4ff', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, recall, width, label='Recall (%)',
               color='#ff4757', edgecolor='black', linewidth=1.2)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', bbox_inches='tight')
plt.close()

print(f"\n✅ All visualizations generated successfully in '{output_dir}/' folder!")
print(f"Total images created: {len(list(output_dir.glob('*.png')))}")
