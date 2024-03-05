import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Load the dataset
df = pd.read_csv('./DatasetmalwareExtrait.csv')

# Preprocess data
features = df.drop('legitimate', axis=1)
targets = df['legitimate']

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Standardize features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Hyperparameter tuning
# Adjust the param_grid to use 'liblinear' for 'l1' penalty or stick with 'l2'
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l2'], 'solver': ['liblinear']}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)

# Print the best parameters
print("Best parameters:", grid.best_params_)

# Train the model with the best parameters
model = LogisticRegression(C=grid.best_params_['C'], penalty=grid.best_params_['penalty'], solver=grid.best_params_['solver'])
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
print("Cross-validation scores:", scores)
print("Average cross-validation score:", scores.mean())
