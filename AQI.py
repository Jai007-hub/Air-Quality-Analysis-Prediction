# Import Required Libraries

import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Regression models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

# Load Dataset

df = pd.read_csv(r"C:\Docs\UPES\Sem - 4\AIML\Project\Air Quality Dataset.csv")
df.head(10000)

# Handle Missing Values

# Fill numeric columns with mean
df.fillna(df.mean(numeric_only=True), inplace=True)
# Fill categorical with mode
for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encoding

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# Separate Features and Targets

# Example (CHANGE according to dataset)
reg_target = df.columns[-2]  # second last column as regression target
clf_target = df.columns[-1]  # last column as classification target
X = df.drop([reg_target, clf_target], axis=1)
y_reg = df[reg_target]
y_clf = df[clf_target]

# Train-Test Split

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

# Feature Scaling

scaler = StandardScaler()
X_train_r = scaler.fit_transform(X_train_r)
X_test_r = scaler.transform(X_test_r)
X_train_c = scaler.fit_transform(X_train_c)
X_test_c = scaler.transform(X_test_c)

# Training Models
# REGRESSION MODELS

reg_models = {
    "Linear": LinearRegression(),
    "Lasso": Lasso(alpha=0.1),
    "Ridge": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
}
reg_results = {}
for name, model in reg_models.items():
    model.fit(X_train_r, y_train_r)
    y_pred = model.predict(X_test_r)
    mse = mean_squared_error(y_test_r, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_r, y_pred)
    reg_results[name] = {"MSE": mse, "RMSE": rmse, "R2": r2}
    print(f"{name} -> MSE:{mse:.2f}, RMSE:{rmse:.2f}, R2:{r2:.2f}")

# CLASSIFICATION MODELS

clf_models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "L1 (Lasso)": LogisticRegression(penalty="l1", solver="liblinear"),
    "L2 (Ridge)": LogisticRegression(penalty="l2"),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(),
}
clf_results = {}
for name, model in clf_models.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict(X_test_c)
    precision = precision_score(y_test_c, y_pred, average="weighted")
    recall = recall_score(y_test_c, y_pred, average="weighted")
    f1_macro = f1_score(y_test_c, y_pred, average="macro")
    f1_micro = f1_score(y_test_c, y_pred, average="micro")
    f1_weighted = f1_score(y_test_c, y_pred, average="weighted")
    clf_results[name] = {
        "Precision": precision,
        "Recall": recall,
        "F1_macro": f1_macro,
        "F1_micro": f1_micro,
        "F1_weighted": f1_weighted,
    }
    print(f"{name}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
    print(
        f"Macro F1: {f1_macro:.3f}, Micro F1: {f1_micro:.3f}, Weighted F1: {f1_weighted:.3f}"
    )
    print("-" * 50)

# Model Comparison (R²)

r2_scores = [reg_results[m]["R2"] for m in reg_results]
plt.figure()
plt.bar(reg_results.keys(), r2_scores)
plt.title("Regression Model Comparison (R2 Score)")
plt.xticks(rotation=45)
plt.show()

# Classification Visualization
# Confusion Matrices

for name, model in clf_models.items():
    y_pred = model.predict(X_test_c)
    cm = confusion_matrix(y_test_c, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.show()

# Model Comparision based on F1 Score

models = list(clf_results.keys())
macro = [clf_results[m]["F1_macro"] for m in models]
micro = [clf_results[m]["F1_micro"] for m in models]
weighted = [clf_results[m]["F1_weighted"] for m in models]
x = np.arange(len(models))
width = 0.25
plt.figure()
plt.bar(x - width, macro, width, label="Macro F1")
plt.bar(x, micro, width, label="Micro F1")
plt.bar(x + width, weighted, width, label="Weighted F1")
plt.xticks(x, models, rotation=45)
plt.ylabel("F1 Score")
plt.title("Comparison of F1 Scores (Macro vs Micro vs Weighted)")
plt.legend()

# ROC-AUC Curve

from sklearn.preprocessing import label_binarize

# Get number of classes
classes = np.unique(y_test_c)
n_classes = len(classes)
# Binarize output
y_test_bin = label_binarize(y_test_c, classes=classes)
plt.figure()
for name, model in clf_models.items():
    # Some models (like SVM) may not have predict_proba unless enabled
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_c)
    else:
        y_prob = model.decision_function(X_test_c)
    # Compute ROC curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} Class {i} (AUC={roc_auc:.2f})")
# Diagonal line
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC-AUC Curve")
plt.legend()
plt.show()
