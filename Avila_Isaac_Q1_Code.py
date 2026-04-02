import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import precision_score, recall_score, classification_report



file_path = "adult.data"
columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
           "hours-per-week", "native-country", "income"]

# Check if the file exists locally
if os.path.exists(file_path):
    # sep=',\s' handles the comma and the whitespace following it in the CSV
    data = pd.read_csv(file_path, names=columns, sep=',\s', engine='python')
    print("File loaded successfully.")
else:
    print(f"Error: {file_path} not found. Ensure the file is in the same folder as this script.")

# Clean missing values (replacing '?' with NaN and dropping)
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

# Encode the target variable (income)
# Removing potential trailing periods often found in the 'adult.test' version of this data
data['income'] = data['income'].str.strip().str.replace('.', '', regex=False)
data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

# One-hot encode categorical features
data_encoded = pd.get_dummies(data)

# Split into features and target
X = data_encoded.drop('income', axis=1)
y = data_encoded['income']

# Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Preprocessing complete. Training shape: {X_train.shape}")


# Initialize models
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
knn = KNeighborsClassifier(n_neighbors=75)
lr = LogisticRegression(max_iter=1000)

# Evaluate each
for clf, label in zip([dt, knn, lr], ['Decision Tree', 'KNN', 'Logistic Regression']):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(f"{label} Accuracy: {accuracy_score(y_test, pred):.4f}")


# Bagging (using Decision Trees)
bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)

# Boosting (AdaBoost and Gradient Boosting)
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Voting (Combining the three base models from Step 2)
voting = VotingClassifier(estimators=[('dt', dt), ('knn', knn), ('lr', lr)], voting='soft')

# Evaluate Ensembles
ensembles = [bagging, ada, gb, voting]
labels = ['Bagging', 'AdaBoost', 'Gradient Boosting', 'Voting']

for clf, label in zip(ensembles, labels):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(f"{label} Accuracy: {accuracy_score(y_test, pred):.4f}")


# Create a dictionary to store our models for easy iteration
all_models = {
    "Decision Tree": dt,
    "KNN": knn,
    "Logistic Regression": lr,
    "Bagging": bagging,
    "AdaBoost": ada,
    "Gradient Boosting": gb,
    "Voting": voting
}

print(f"{'Model':<20} | {'Precision':<10} | {'Recall':<10}")
print("-" * 45)

for name, model in all_models.items():
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    # Using pos_label=1 because we want to track the ">50K" class
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"{name:<20} | {precision:<10.4f} | {recall:<10.4f}")