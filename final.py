import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Optional boosters (install if missing):
#   python3 -m pip install xgboost lightgbm
try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:
    LGBMClassifier = None

# Load the featured dataset
df = pd.read_csv('featured_mindtune.csv')

# Define features and target
# Exclude identifiers and the string label
drop_cols = ['session_id', 'participant_id', 'timestamp_ms', 'label_3class', 'label_encoded']
X = df.drop(columns=drop_cols)
y = df['label_encoded']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to test
models = {
    # sklearn 1.8+ removed `multi_class`; multiclass uses multinomial loss with solver lbfgs/saga by default.
    "Logistic Regression": LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

if XGBClassifier is not None:
    models["XGBoost"] = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

if LGBMClassifier is not None:
    models["LightGBM"] = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

results = []
trained_models = {}

for name, model in models.items():
    # Use scaled data for LogReg, raw data for others
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    trained_models[name] = model
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append({"Model": name, "Accuracy": acc, "F1-Score": f1})

results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)

print("Model Comparison Results:")
print(results_df)

# Plotting the results
plt.figure(figsize=(10, 6))
sns.barplot(x='F1-Score', y='Model', data=results_df, palette='viridis')
plt.title('Model Performance Comparison (F1-Score)')
plt.xlim(0, 1.0)
plt.tight_layout()
plt.savefig('model_comparison.png')

# Detailed report for the best model
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]

if best_model_name == "Logistic Regression":
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

print(f"\nDetailed Classification Report for {best_model_name} (Best Model):")
print(classification_report(y_test, y_pred_best, target_names=['calm', 'neutral', 'stressed']))

# Feature Importance for the best tree-based model (if applicable)
if best_model_name in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
    importances = best_model.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 8))
    feat_importances.sort_values().plot(kind='barh', color='teal')
    plt.title(f'Top 15 Feature Importances ({best_model_name})')
    plt.tight_layout()
    plt.savefig('feature_importance.png')