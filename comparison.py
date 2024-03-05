import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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


# Function to train and evaluate a model
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    avg_cv_score = cv_scores.mean()
    return accuracy, f1, avg_cv_score


# Train and evaluate Logistic Regression
log_reg = LogisticRegression(solver='liblinear', random_state=42)
log_reg_accuracy, log_reg_f1, log_reg_cv_score = train_and_evaluate(log_reg, X_train, X_test, y_train, y_test)

# Train and evaluate Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_accuracy, rf_f1, rf_cv_score = train_and_evaluate(rf, X_train, X_test, y_train, y_test)

# Train and evaluate SVM
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm_accuracy, svm_f1, svm_cv_score = train_and_evaluate(svm, X_train, X_test, y_train, y_test)

# Compare models
print("Logistic Regression:")
print(f"Accuracy: {log_reg_accuracy}, F1 Score: {log_reg_f1}, Average CV Score: {log_reg_cv_score}")
print("\nRandom Forest:")
print(f"Accuracy: {rf_accuracy}, F1 Score: {rf_f1}, Average CV Score: {rf_cv_score}")
print("\nSVM:")
print(f"Accuracy: {svm_accuracy}, F1 Score: {svm_f1}, Average CV Score: {svm_cv_score}")

# Determine the best model
best_model = max([('Logistic Regression', log_reg_f1), ('Random Forest', rf_f1), ('SVM', svm_f1)], key=lambda x: x[1])
print(f"\nThe best model is: {best_model[0]} with an F1 score of {best_model[1]}")
