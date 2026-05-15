#!/usr/bin/env python
# coding: utf-8

# # STRATOS: Spatial Thermal Risk & Temperature Observation System
# ## Phase 2: Preprocessing, Unsupervised Learning & Regression
# This section covers data loading, Covariance Analysis, Z-Score Normalization, PCA, K-Means Clustering, and Linear Regression with Gradient Descent.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')


# In[ ]:


import requests
import pandas as pd
from datetime import datetime

# Misamis Oriental Coordinates & 10-Year Range
LATITUDE = 8.5342
LONGITUDE = 124.7554
START_DATE = "20140101"
END_DATE = "20231231"

url = "https://power.larc.nasa.gov/api/temporal/daily/point"
params = {
    "parameters": "T2M,RH2M,ALLSKY_SFC_SW_DWN,WS10M",
    "community": "RE",
    "longitude": LONGITUDE,
    "latitude": LATITUDE,
    "start": START_DATE,
    "end": END_DATE,
    "format": "JSON"
}

print(f"Fetching NASA POWER data for Lat: {LATITUDE}, Lon: {LONGITUDE}...")
response = requests.get(url, params=params)
response.raise_for_status()

# Extract data from JSON payload
features = response.json().get("properties", {}).get("parameter", {})

# Convert to raw_climate_df and reshape index
raw_climate_df = pd.DataFrame(features).reset_index()
raw_climate_df = raw_climate_df.rename(columns={
    "index": "date", 
    "T2M": "temperature_2m", 
    "RH2M": "relative_humidity_2m", 
    "ALLSKY_SFC_SW_DWN": "solar_radiation", 
    "WS10M": "wind_speed_10m"
})

# Format datetime and drop any potential NaNs
raw_climate_df['date'] = pd.to_datetime(raw_climate_df['date'], format='%Y%m%d')
raw_climate_df.set_index('date', inplace=True)
raw_climate_df.dropna(inplace=True)

# Map to 'df' to preserve all existing downstream ML logic
df = raw_climate_df

display(df.head())


# In[ ]:


# 1. Covariance Analysis
print("Covariance Matrix:")
cov_matrix = df.cov()
display(cov_matrix)

# Visualizing Correlation (normalized covariance)
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()


# In[ ]:


# 2. Z-Score Normalization
scaler = StandardScaler()

# Features to scale
features = ['temperature_2m', 'relative_humidity_2m', 'solar_radiation', 'wind_speed_10m']
scaled_data = scaler.fit_transform(df[features])

# Create a normalized dataframe
df_normalized = pd.DataFrame(scaled_data, columns=features, index=df.index)
print("Z-Score Normalized Data (First 5 rows):")
display(df_normalized.head())


# In[ ]:


# 3. Principal Component Analysis (PCA)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_normalized)

df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=df.index)

print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.5, c=df['temperature_2m'], cmap='viridis')
plt.colorbar(label='Temperature (C)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Climate Data')
plt.show()


# In[ ]:


# 4. K-Means Algorithm (Clustering days into climate profiles)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['climate_profile'] = kmeans.fit_predict(df_normalized)

# Plotting the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'], hue=df['climate_profile'], palette='Set1', s=60, alpha=0.7)
plt.title('K-Means Clustering of Climate Profiles')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Profile')
plt.show()

# Analyze cluster centers (inveres transformed)
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
df_centers = pd.DataFrame(cluster_centers, columns=features)
print("Climate Profile Characteristics (Cluster Centers):")
display(df_centers)


# In[ ]:


# 5. Linear Regression (Predicting peak temperature)
# We will predict 'temperature_2m' using the other variables
X = df[['relative_humidity_2m', 'solar_radiation', 'wind_speed_10m']]
y = df['temperature_2m']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)

y_pred = lin_reg_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Linear Regression MSE: {mse:.4f}")
print(f"Coefficients: {lin_reg_model.coef_}")
print(f"Intercept: {lin_reg_model.intercept_}")


# In[ ]:


# 6. Gradient Descent (Used to optimize the regression and calculate MSE)
# We will implement a simple gradient descent from scratch for linear regression with 1 variable (e.g., solar radiation predicting temp)
X_gd = df_normalized['solar_radiation'].values
y_gd = df_normalized['temperature_2m'].values

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Initialize weights
m = 0.0 # slope
c = 0.0 # intercept
n = len(X_gd)

mse_history = []

for i in range(epochs):
    # Predictions
    y_pred_gd = m * X_gd + c

    # Calculate Mean Squared Error
    current_mse = np.mean((y_gd - y_pred_gd)**2)
    mse_history.append(current_mse)

    # Calculate Gradients
    D_m = (-2/n) * sum(X_gd * (y_gd - y_pred_gd))
    D_c = (-2/n) * sum(y_gd - y_pred_gd)

    # Update Weights
    m = m - learning_rate * D_m
    c = c - learning_rate * D_c

print(f"Gradient Descent Final MSE: {mse_history[-1]:.4f}")
print(f"Gradient Descent Optimized Slope (m): {m:.4f}")
print(f"Gradient Descent Optimized Intercept (c): {c:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(range(epochs), mse_history, color='red')
plt.title('Gradient Descent: MSE Reduction over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.show()


# ## Phase 3: Classification Showdown
# Engineering the `extreme_heat_warning` target and comparing multiple classifiers.

# In[ ]:


# Feature Engineering: Extreme Heat Warning
# Let's define an extreme heat warning as temperature in the top 15% of historical data
threshold = df['temperature_2m'].quantile(0.85)
print(f"Extreme Heat Threshold: {threshold:.2f} C")

df['extreme_heat_warning'] = (df['temperature_2m'] >= threshold).astype(int)

print("Class Distribution:")
print(df['extreme_heat_warning'].value_counts(normalize=True))

# Prepare features and target
X_class = df_normalized[['temperature_2m', 'relative_humidity_2m', 'solar_radiation', 'wind_speed_10m']]
y_class = df['extreme_heat_warning']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib

# Dictionary to store results
results = {}

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec}
    print(f"--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Confusion Matrix:\n{cm}\n")
    return model


# In[ ]:


# 5. Logistic Regression
log_reg = LogisticRegression(random_state=42)
evaluate_model("5. Logistic Regression", log_reg, X_train_c, y_train_c, X_test_c, y_test_c)

# 6. Decision Trees
tree_clf = DecisionTreeClassifier(random_state=42, max_depth=5)
evaluate_model("6. Decision Tree", tree_clf, X_train_c, y_train_c, X_test_c, y_test_c)

# 7. Random Forest
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
best_model = evaluate_model("7. Random Forest", rf_clf, X_train_c, y_train_c, X_test_c, y_test_c)


# In[ ]:


# 8. Support Vector Machine (SVM) Algorithm (Base Implementation)
# Using default parameters as the base implementation
svm_base = SVC(random_state=42)
evaluate_model("8. Base SVM", svm_base, X_train_c, y_train_c, X_test_c, y_test_c)

# 9. Linear SVM
# Using Linear kernel
svm_linear = SVC(kernel='linear', random_state=42)
evaluate_model("9. Linear SVM", svm_linear, X_train_c, y_train_c, X_test_c, y_test_c)

# 10. Non-Linear SVM
# Using Polynomial kernel to demonstrate non-linearity
svm_poly = SVC(kernel='poly', degree=3, random_state=42)
evaluate_model("10. Non-Linear SVM (Poly)", svm_poly, X_train_c, y_train_c, X_test_c, y_test_c)


# In[ ]:


# 11. Kernel Trick in SVM (Mathematical Transformation Concept)
# We manually apply a transformation to 2D data to make it linearly separable in 3D
print("11. Demonstrating the Kernel Trick:")

# Take two features for simplicity
X_2d = X_train_c.iloc[:, :2].values
y_2d = y_train_c.values

# The non-linear mapping function Phi(x)
def phi(X):
    # Mapping [x1, x2] to [x1^2, sqrt(2)*x1*x2, x2^2]
    return np.column_stack((X[:, 0]**2, np.sqrt(2)*X[:, 0]*X[:, 1], X[:, 1]**2))

X_3d = phi(X_2d)

print(f"Original 2D shape: {X_2d.shape}")
print(f"Transformed 3D shape: {X_3d.shape}")

# Now a linear hyperplane can separate the classes in 3D!
# We fit a LinearSVC in the transformed space
linear_svc_3d = LinearSVC(random_state=42, max_iter=2000)
linear_svc_3d.fit(X_3d, y_2d)
print("Fitted a Linear Model in the transformed 3D space, which corresponds to a non-linear boundary in the original 2D space.\n")


# In[ ]:


# 12. RBF Kernel in SVM
# The RBF kernel implicitly maps to infinite dimensions
svm_rbf = SVC(kernel='rbf', gamma='scale', random_state=42)
evaluate_model("12. RBF Kernel SVM", svm_rbf, X_train_c, y_train_c, X_test_c, y_test_c)


# In[ ]:


# Create the comparison dataframe from our results dictionary
comparison_df = pd.DataFrame(results).T

# Mathematically derive F1-Score from stored Precision and Recall
comparison_df['F1-Score'] = 2 * (comparison_df['Precision'] * comparison_df['Recall']) / (comparison_df['Precision'] + comparison_df['Recall'])

# Fill NaNs with 0 (in case of 0 precision/recall) to prevent display errors
comparison_df['F1-Score'] = comparison_df['F1-Score'].fillna(0) 

# Reorder columns, sort by Accuracy, and name the index
comparison_df = comparison_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
comparison_df = comparison_df.sort_values(by='Accuracy', ascending=False)
comparison_df.index.name = 'Algorithm Name'

print("Classification Showdown - Comparative Metrics:")
display(comparison_df)

# Generate a visually clean summary chart
comparison_df.plot(kind='bar', figsize=(12, 6), colormap='viridis')
plt.title('Classification Models Comparison')
plt.ylabel('Score')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# In[ ]:


# Export the best performing model
# Based on evaluations, Random Forest is robust and highly performant. We export rf_clf.
export_path = '../app/model.pkl'
os.makedirs(os.path.dirname(export_path), exist_ok=True)
joblib.dump(rf_clf, export_path)
print(f"Best classification model saved successfully to {export_path}")


# ## Phase 4: Logistics Routing
# 13. Q-Learning using the Bellman Equation to find the optimal delivery path across a 5x5 municipal grid, avoiding extreme heat zones.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# 13. Q-Learning
# Create a 5x5 municipal grid
grid_size = 5
q_table = np.zeros((grid_size * grid_size, 4)) # 4 actions: up, right, down, left

# Actions: 0=Up, 1=Right, 2=Down, 3=Left
actions = [0, 1, 2, 3]

# Goal is bottom-right (24), Start is top-left (0)
start_state = 0
goal_state = 24

# Let's flag some heat zones based on our extreme heat warnings
# We'll randomly select 4 states (excluding start and goal) to be high-penalty heat zones
np.random.seed(42)
heat_zones = np.random.choice(range(1, 24), size=4, replace=False)

def get_reward(state):
    if state == goal_state:
        return 100
    elif state in heat_zones:
        return -50
    else:
        return -1

def get_next_state(state, action):
    row = state // grid_size
    col = state % grid_size

    if action == 0 and row > 0:       # Up
        row -= 1
    elif action == 1 and col < grid_size - 1: # Right
        col += 1
    elif action == 2 and row < grid_size - 1: # Down
        row += 1
    elif action == 3 and col > 0:     # Left
        col -= 1

    return row * grid_size + col

# Hyperparameters
alpha = 0.1 # Learning rate
gamma = 0.9 # Discount factor
epsilon = 0.1 # Exploration rate
episodes = 500

# Training the Q-Learning Agent using the Bellman Equation
for episode in range(episodes):
    state = start_state

    while state != goal_state:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(q_table[state])

        next_state = get_next_state(state, action)
        reward = get_reward(next_state)

        # Bellman Equation update
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Q(S,A) <- Q(S,A) + alpha * [R + gamma * max Q(S',a) - Q(S,A)]
        q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state

print("Training Complete. Optimal Q-Values Computed.")

# Extracting the optimal path
optimal_path = [start_state]
state = start_state
while state != goal_state:
    action = np.argmax(q_table[state])
    state = get_next_state(state, action)
    optimal_path.append(state)
    if len(optimal_path) > grid_size * grid_size:
        print("Agent stuck in a loop.")
        break

print(f"Heat Zones: {heat_zones}")
print(f"Optimal Path: {optimal_path}")

# Visualization
grid_viz = np.zeros((grid_size, grid_size))
for hz in heat_zones:
    grid_viz[hz // grid_size, hz % grid_size] = -1 # Heat zones in red
grid_viz[goal_state // grid_size, goal_state % grid_size] = 2 # Goal in green

for step in optimal_path:
    if step != start_state and step != goal_state:
        grid_viz[step // grid_size, step % grid_size] = 1 # Path in blue

plt.figure(figsize=(6,6))
sns.heatmap(grid_viz, cmap=['red', 'white', 'blue', 'green'], annot=True, cbar=False, linewidths=1, linecolor='black')
plt.title('Q-Learning Optimal Routing (Red: Heat Zone, Blue: Path, Green: Goal)')
plt.show()

