import pandas as pd
import pathlib
import time
from sklearn.cluster import KMeans, Birch
from sklearn.preprocessing import StandardScaler

# Define the dataset path
input_path = pathlib.Path(__file__).parent.parent / 'data' / 'data.csv'

# Check if the file exists
if not input_path.exists():
    raise FileNotFoundError(f'The file {input_path} does not exist.')

start = time.time()

print('Reading and dropping columns...\n')

# Read the CSV file into a DataFrame
df = pd.read_csv(input_path)

# Columns to drop
columns_to_drop = [
    'Timestamp', 'Bwd Segment Size Avg', 'Fwd Packets/s', 'Packet Length Max',
    'Fwd Packet Length Mean', 'Flow IAT Max', 'Packet Length Mean',
    'Bwd Packet/Bulk Avg', 'Active Min', 'ACK Flag Count', 'Active Max',
    'Idle Min', 'Bwd IAT Total', 'Fwd Act Data Pkts', 'Fwd IAT Max',
    'Average Packet Size', 'Fwd Packet Length Min', 'Subflow Bwd Packets',
    'Bwd Packet Length Std', 'Bwd IAT Max', 'Fwd IAT Mean', 'Idle Mean',
    'Packet Length Variance'
]

df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

print('Columns dropped.\nPerforming stratified sampling by Label...')

# Stratified sampling - 0.5% from each class in 'Label'
sample_frac = 0.015
if 'Label' not in df.columns:
    raise ValueError("Label column not found in the DataFrame.")

df_stratified = df.groupby(['Label', 'Traffic Type'], group_keys=False).sample(frac=sample_frac, random_state=42)


# Save stratified sample
output_path = pathlib.Path(__file__).parent.parent / 'data' / 'data_stratified.csv'
df_stratified.to_csv(output_path, index=False, encoding='utf-8-sig')

# Normalize features for clustering
print('Performing KMeans clustering on the original dataset...')
features = df.drop(columns=['Traffic Type'], errors='ignore').select_dtypes(include=['number'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# KMeans clustering
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Custom sampling: 50% from Cluster 1, 0.5% from others
print('Sampling: 50% from Cluster 1, 0.5% from others...')
samples = []

for cluster_id, group in df.groupby('Cluster'):
    if cluster_id == 1:
        samples.append(group.sample(frac=0.5, random_state=42))  # take half of cluster 1
    else:
        samples.append(group.sample(frac=sample_frac, random_state=42))  # 0.5% from others

df_sample_custom = pd.concat(samples)

# Save the custom sampled DataFrame from KMeans
output_path_clustered = pathlib.Path(__file__).parent.parent / 'data' / 'data_kmeans_custom.csv'
df_sample_custom.to_csv(output_path_clustered, index=False, encoding='utf-8-sig')

# Birch clustering on stratified dataset
print('Performing Birch clustering on the stratified dataset (from Label)...')

features_sample = df_stratified.drop(columns=['Label', 'Cluster'], errors='ignore').select_dtypes(include='number')
X_sample_scaled = scaler.fit_transform(features_sample)

birch = Birch(n_clusters=4)
df_stratified['Birch_Cluster'] = birch.fit_predict(X_sample_scaled)

# Custom sampling: 50% from Birch_Cluster 2, 0.5% from others
print("Custom sampling from Birch clusters: 50% from Cluster 2, 0.5% from others...")

birch_samples = []

for cluster_id, group in df_stratified.groupby('Birch_Cluster'):
    if cluster_id == 2:
        birch_samples.append(group)  # 50% from cluster 2
    else:
        birch_samples.append(group.sample(frac=0.5, random_state=42))  # 0.5% from others

df_stratified_birch = pd.concat(birch_samples)

# Save Birch result
output_path_birch = pathlib.Path(__file__).parent.parent / 'data' / 'data_birch_custom.csv'
df_stratified_birch.to_csv(output_path_birch, index=False, encoding='utf-8-sig')

end = time.time()
print("All clustering and sampling steps complete.")
print(f"Total time taken: {end - start:.2f} seconds.")
