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

print('Dropping columns...\n')

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

print('Columns dropped.\nSampling...')

# Stratified sampling - 10% from each class in 'Label'
sample_frac = 0.001
if 'Label' not in df.columns:
    raise ValueError("Label column not found in the DataFrame.")

output_path = pathlib.Path(__file__).parent.parent / 'data' / 'data_stratified.csv'
df_stratified = df.groupby('Label', group_keys=False).sample(frac=sample_frac, random_state=42)
df_stratified.to_csv(output_path, index=False, encoding='utf-8-sig')

# Normalization for the clustering
print('Performing KMeans clustering on the original dataset...')
features = df.drop(columns=['Traffic Type'], errors='ignore').select_dtypes(include=['number'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Clustering with KMeans
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Stratified Sampling 10% from each KMeans cluster 
df_sample = df.groupby('Cluster', group_keys=False).sample(frac=sample_frac, random_state=42)

# Save KMeans output
output_path_clustered = pathlib.Path(__file__).parent.parent / 'data' / 'data_kmeans.csv'
df_sample.to_csv(output_path_clustered, index=False, encoding='utf-8-sig')

# Birch clustering on the stratified KMeans sample ===
print('Performing Birch clustering on the sampled dataset...')

features_sample = df_stratified.drop(columns=['Label', 'Cluster'], errors='ignore').select_dtypes(include='number')
X_sample_scaled = scaler.fit_transform(features_sample)

birch = Birch(n_clusters=5)
df_stratified['Birch_Cluster'] = birch.fit_predict(X_sample_scaled)

# sampling 10% from each Birch_Cluster 
df_stratified_birch = df_stratified.groupby('Birch_Cluster', group_keys=False).sample(frac=0.10, random_state=42)

# Save Birch output 
output_path_birch = pathlib.Path(__file__).parent.parent / 'data' / 'data_birch.csv'
df_stratified_birch.to_csv(output_path_birch, index=False, encoding='utf-8-sig')

end = time.time()
print("All clustering and sampling steps complete.")
print(f"Total time taken: {end - start:.2f} seconds.")
