import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction import FeatureHasher

# Choose the dataset
file_choice = input('Type 1 for data_stratified.csv, 2 for data_kmeans.csv, or 3 for data_birch.csv: ')
if file_choice == '1':
    file_name = 'data_stratified.csv'
elif file_choice == '2':
    file_name = 'data_kmeans_custom.csv'
elif file_choice == '3':
    file_name = 'data_birch_custom.csv'
else:
    raise ValueError('Invalid input. Please enter 1, 2, or 3.')

input_path = pathlib.Path(__file__).parent.parent / 'data' / file_name
if not input_path.exists():
    raise FileNotFoundError(f'The file {input_path} does not exist.')

print('Loading dataset...')
df = pd.read_csv(input_path)

# Select the label to base classification on
class_label = input('Type 1 for class label "Label" (binary), or 2 for class label "Traffic Type" (multiclass): ')
if class_label == '1':
    label = 'Label'
elif class_label == '2':
    label = 'Traffic Type'
else:
    raise ValueError('Invalid input. Please enter 1 or 2.')

if label not in df.columns:
    raise ValueError(f'The class label "{label}" does not exist in the DataFrame.')

# Choose classifier type
classifier_choice = input('Type 1 for MLP, 2 for SVM: ')
if classifier_choice not in ['1', '2']:
    raise ValueError('Invalid input. Please enter 1 or 2.')

# Extract the label column
y = df[label]

# Convert string labels to numeric (important for sklearn models)
print('Encoding labels...')
le = LabelEncoder()
y = le.fit_transform(y)
 
# Extract the feature columns
X = df.drop(columns=[label])

# Separate numerical and categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=[np.number]).columns

# Encode categorical columns using FeatureHasher (MLP and SVM take numeric input)
# (MurmurHash-based for speed and memory efficiency)
print('Encoding categorical features...')
hasher = FeatureHasher(n_features=128, input_type='string')
hashed_features = hasher.transform(X[categorical_cols].astype(str).to_dict(orient='records'))
X_hashed = pd.DataFrame(hashed_features.toarray())

# Standard scaling for numerical columns
print('Scaling numerical features...')
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X[numerical_cols]), columns=numerical_cols)

# Combine processed numerical and hashed categorical features
X_final = pd.concat([X_scaled.reset_index(drop=True), X_hashed.reset_index(drop=True)], axis=1)
X_final.columns = X_final.columns.astype(str)  # Ensure all column names are strings

# Ensure all values are numeric and handle NaNs
print('Converting features to numeric...')
X_final = X_final.apply(pd.to_numeric, errors='coerce')
if X_final.isnull().values.any():
    print("Warning: NaNs found in features. Filling with 0.")
    X_final = X_final.fillna(0)

# Perform train-test split (default)
print('Splitting dataset into training and testing sets...')
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Neural Network Classifier (MLP)
if classifier_choice == '1':
    
    # Since the dataset is large, the parameters below focus on speed
    # Same parameters for both models (binary and multiclass), apart from neuron and layer numbers
    common_params = {
        'activation': 'relu',            # Default, good for speed
        'solver': 'adam',                # Default, good for speed
        'alpha': 0.0005,                 # A little higher than default (0.0001) to reduce overfitting
        'batch_size': 512,               # Large batch for fast training 
        'learning_rate': 'constant',     # Default learning rate for speed
        'learning_rate_init': 0.005,     # A little higher than default (0.0001) for speed
        'max_iter': 1000,                # A little higher than default for better results (default 200)
        'shuffle': True,                 # Default shuffle
        'random_state': 42,              # Default seed
        'tol': 1e-3,                     # Early convergence tolerance for speed (default 1e-4)
        'early_stopping': True,          # Stop training if validation score doesn't improve
        'validation_fraction': 0.1,      # 10% of training data for validation (default)
        'n_iter_no_change': 5,           # Stop after 5 epochs without improvement (default)
        'verbose': True                  # Print the progress
    }

    if label == 'Label':
        # Binary classification MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=(64,), # One layer for speed (binary)
            **common_params
        )
    else:
        # Multiclass classification MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32), # Two layers for better decision making (multiclass)
            **common_params
        )

    # Fit the model
    print('Training MLP classifier...')
    mlp.fit(X_train, y_train)

    # Evaluate
    print('Evaluating MLP classifier...')
    y_pred = mlp.predict(X_test)

# Support Vector Machine Classifier (SVM)
elif classifier_choice == '2':
    # Parameters focused on speed for large datasets
    common_params = {
        'kernel': 'linear',           # Linear kernel for speed (default 'rbf')
        'C': 1.0,                     # Regularization parameter (default 1.0)
        'tol': 1e-3,                  # Stopping tolerance for optimization (default 1e-3)
        'max_iter': 1000,             # Limit iterations for speed (default -1)
        'class_weight': 'balanced',   # Handle class imbalance 
        'shrinking': False,           # Disable shrinking heuristic 
        'cache_size': 500,            # Kernel cache size in MB (default=200)
        'random_state': 42,           # Seed for reproducibility
        'verbose': True               # Print progress
    }

    if label == 'Label':
        # Binary classification SVM
        svm = SVC(
            **common_params
        )
    else:
        # Multiclass classification SVM
        svm = SVC(
            decision_function_shape='ovr',  # One-vs-rest for speed
            **common_params
        )

    # Fit the model
    print('Training SVM classifier...')
    svm.fit(X_train, y_train)
    
    # Evaluate
    print('Evaluating SVM classifier...')
    y_pred = svm.predict(X_test)

else:
    raise ValueError('Invalid classifier choice.')

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
