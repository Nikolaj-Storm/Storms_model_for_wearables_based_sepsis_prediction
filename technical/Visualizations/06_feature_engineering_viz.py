# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  06_feature_engineering_viz.py
#  Stage: 6 - Visualization
#
#  PURPOSE
#    Engineers rolling-std and 4-hour diff trend features on the 1-hour
#    cohort, then produces three academic figures: physiological KDE
#    distributions (sepsis vs control), a single-patient heart-rate
#    trajectory (continuous vs 4-hour baseline), and a feature-vs-target
#    correlation heatmap.
#
#  INPUTS
#    dataset_b_1h_train.parquet
#  OUTPUTS
#    01_viz_distributions.png
#    02_viz_trajectory.png
#    03_viz_correlations.png
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    input parquet     -  dataset_b_1h_train.parquet (1-hour cohort)
#    target column     -  'is_sepsis'
#    figure outputs    -  01_viz_distributions.png, 02_viz_trajectory.png,
#                         03_viz_correlations.png (all dpi=300)
#
#  REQUIRES: pandas, numpy, matplotlib, seaborn
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("-------------------------------------------------------------------------")
print("Phase 3: Automated Feature Engineering & Academic Visualizations")
print("-------------------------------------------------------------------------\n")

print("Loading Dataset B (1-Hour) Training data from Parquet...")
if not os.path.exists('dataset_b_1h_train.parquet'):  # EDIT: input 1-hour cohort parquet path
    print("Error: dataset_b_1h_train.parquet not found. Ensure 02_dataset_preparation.py ran completely.")
    exit(1)

df_1h = pd.read_parquet('dataset_b_1h_train.parquet')  # EDIT: input 1-hour cohort parquet path
# Ensure stay_id is available as a column instead of only in index, or vice versa
if 'stay_id' not in df_1h.columns and df_1h.index.name != 'stay_id':
    df_1h = df_1h.reset_index()

# Set index to standard time format for rolling functions
df_1h['charttime'] = pd.to_datetime(df_1h.index if df_1h.index.name == 'charttime' else df_1h['charttime'])
if df_1h.index.name != 'charttime':
    df_1h.set_index('charttime', inplace=True)

# 1. Feature Engineering
print("\n[1/3] Engineering Dynamic Time-Series Features (Rolling & SLOPE)...")
def engineer_features(data):
    vitals_cols = ['heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp']
    grouped = data.groupby('stay_id')

    # Calculate physiological instability (Standard Deviation over 4h)
    # Calculate deterioration velocity (Rate-of-Change / Diff over 4h)
    for v in vitals_cols:
        data[f'{v}_std_4h'] = grouped[v].transform(lambda x: x.rolling(window=4, min_periods=1).std())
        data[f'{v}_diff_4h'] = grouped[v].diff(periods=4)

    data = data.fillna(0) # Default fill for edges where variance/diff cannot be established
    return data

df_engineered = engineer_features(df_1h)

# 2. Visualization: Sepsis vs Non-Sepsis KDE Distributions
print("[2/3] Generating Visualization 1: Physiological Distributions (Sepsis vs Non-Sepsis)...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
vitals_labels = {
    'heart_rate': 'Heart Rate (bpm)', 'resprate': 'Respiratory Rate (bpm)',
    'spo2': 'SpO2 (%)', 'temp_c': 'Temperature (°C)',
    'sbp': 'Systolic BP (mmHg)', 'dbp': 'Diastolic BP (mmHg)'
}

# Explicitly map the 1 to "Sepsis" and 0 to "Control" for plotting clarity
plot_df = df_engineered.copy()
plot_df['Sepsis Status'] = plot_df['is_sepsis'].map({1: 'Sepsis Patient', 0: 'Control (Non-Sepsis)'})  # EDIT: target/label column

for ax, (vital, label) in zip(axes.flatten(), vitals_labels.items()):
    sns.kdeplot(data=plot_df, x=vital, hue='Sepsis Status', common_norm=False, fill=True, alpha=0.4, ax=ax, palette=['#e74c3c', '#2c3e50'])
    ax.set_title(f'Distribution of {label}')
    ax.set_xlabel('')

plt.suptitle("Raw Physiological Distributions (1-Hour Resolution)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('01_viz_distributions.png', dpi=300)  # EDIT: figure output filename and DPI
plt.close()

# 3. Visualization: Time-Series Trajectory for a Single Sepsis Patient
print("[3/3] Generating Visualization 2: Continuous Patient Heart Rate Trajectory...")
sepsis_patients = plot_df[plot_df['is_sepsis'] == 1]['stay_id'].unique()
if len(sepsis_patients) > 0:
    example_patient = sepsis_patients[np.random.randint(0, min(100, len(sepsis_patients)))] # Pick a random patient
    patient_data = plot_df[plot_df['stay_id'] == example_patient].reset_index()

    plt.figure(figsize=(14, 6))
    plt.plot(patient_data['charttime'], patient_data['heart_rate'], marker='o', label='1-Hour Wearable Measurement', color='#e74c3c', linewidth=2)
    plt.plot(patient_data['charttime'], patient_data['heart_rate'].rolling(4).mean(), linestyle='--', label='4-Hour "Standard Care" Baseline', color='#2c3e50', linewidth=3)

    # Shade the 4-hour standard deviation block to show instability
    plt.fill_between(patient_data['charttime'],
                     patient_data['heart_rate'] - patient_data['heart_rate_std_4h'],
                     patient_data['heart_rate'] + patient_data['heart_rate_std_4h'],
                     color='#e74c3c', alpha=0.15, label='Volatility (4-Hour Std Dev)')

    plt.title(f'Continuous vs. Intermittent Heart Rate Trajectory Prior to Sepsis (Stay ID: {example_patient})', fontsize=14, fontweight='bold')
    plt.xlabel('Time in ICU', fontsize=12)
    plt.ylabel('Heart Rate (bpm)', fontsize=12)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('02_viz_trajectory.png', dpi=300)  # EDIT: figure output filename and DPI
    plt.close()

# 4. Visualization: Feature Correlation Heatmap
print("[4/4] Generating Visualization 3: Feature Architecture Correlation Heatmap...")
corr_cols = ['is_sepsis', 'heart_rate', 'heart_rate_std_4h', 'heart_rate_diff_4h',
             'resprate', 'resprate_std_4h',
             'temp_c', 'temp_c_diff_4h',
             'sbp', 'sbp_diff_4h']

# Rename columns for the actual heatmap text
display_names = {
    'is_sepsis': 'Target (Sepsis)',
    'heart_rate': 'HR (Raw)',
    'heart_rate_std_4h': 'HR Volatility (Std Dev)',
    'heart_rate_diff_4h': 'HR Slope (Velocity)',
    'resprate': 'Resp Rate (Raw)',
    'resprate_std_4h': 'Resp Volatility',
    'temp_c': 'Temp (Raw)',
    'temp_c_diff_4h': 'Temp Slope',
    'sbp': 'Systolic BP',
    'sbp_diff_4h': 'BP Slope'
}

corr_df = df_engineered[corr_cols].rename(columns=display_names)
corr = corr_df.corr()

plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'})
plt.title('Feature Matrix: How the Engineered "Trend" Features Correlate to Sepsis Development', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('03_viz_correlations.png', dpi=300)  # EDIT: figure output filename and DPI
plt.close()

print("\n✅ Engineered features constructed and Thesis Visualizations successfully exported to the 'anti bachelor' folder!")
