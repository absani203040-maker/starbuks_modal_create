# Starbucks Rewards: EDA + Decision Tree Classification
# Run this entire script in ONE Kaggle notebook cell.

import os
import glob
import ast
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_DIR = None  # Set this to your dataset folder if automatic search fails.

def locate_file(filename):
    paths = []
    if DATA_DIR:
        paths += glob.glob(os.path.join(DATA_DIR, "**", filename), recursive=True)
    paths += glob.glob(os.path.join("/kaggle/input", "**", filename), recursive=True)
    paths += glob.glob(os.path.join(".", "**", filename), recursive=True)
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise FileNotFoundError(
            "Cannot find " + filename +
            ". Add the dataset to Kaggle Input or set DATA_DIR to the folder containing the CSV files."
        )
    print("Reading:", paths[0])
    return paths[0]

portfolio = pd.read_csv(locate_file("portfolio.csv"))
profile = pd.read_csv(locate_file("profile.csv"))
transcript = pd.read_csv(locate_file("transcript.csv"))

print("\\n=== DATASET OVERVIEW ===")
for name, frame in [("portfolio", portfolio), ("profile", profile), ("transcript", transcript)]:
    print("\\n", name.upper(), "shape:", frame.shape)
    print("Columns:", frame.columns.tolist())
    display(frame.head())
    print("Duplicate rows:", frame.duplicated().sum())
    missing = frame.isna().sum()
    print("Missing values:")
    display(missing[missing > 0].sort_values(ascending=False).to_frame("count"))

# Simple dataset size chart
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, frame) in zip(
    axes, [("Portfolio", portfolio), ("Profile", profile), ("Transcript", transcript)]
):
    ax.bar(["Rows", "Columns"], [frame.shape[0], frame.shape[1]])
    ax.set_title(name + " dimensions")
plt.tight_layout()
plt.show()

# Event, demographic and numeric EDA
if "event" in transcript.columns:
    plt.figure(figsize=(10, 4))
    transcript["event"].value_counts().plot(kind="bar")
    plt.title("Transcript event counts")
    plt.xlabel("Event")
    plt.ylabel("Count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()

for column, title in [("gender", "Customer gender"), ("age", "Customer age"), ("income", "Customer income")]:
    if column in profile.columns:
        values = profile[column]
        plt.figure(figsize=(8, 4))
        if column == "gender":
            values.fillna("Missing").value_counts().plot(kind="bar")
            plt.ylabel("Customers")
        else:
            pd.to_numeric(values, errors="coerce").dropna().plot(kind="hist", bins=25)
            plt.ylabel("Frequency")
        plt.title(title + " distribution")
        plt.tight_layout()
        plt.show()

# Workflow flowchart
plt.figure(figsize=(12, 2.5))
plt.axis("off")
steps = [("Load data", 0.08), ("Clean data", 0.28), ("Create features", 0.49),
         ("Train tree", 0.70), ("Evaluate", 0.90)]
for label, x in steps:
    plt.text(x, 0.5, label, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.6", edgecolor="black", facecolor="white"))
for i in range(len(steps) - 1):
    plt.annotate("", xy=(steps[i + 1][1] - 0.07, 0.5),
                 xytext=(steps[i][1] + 0.07, 0.5),
                 arrowprops=dict(arrowstyle="->"))
plt.title("Machine Learning Workflow")
plt.tight_layout()
plt.show()

print("\\n=== PREPARE TRANSCRIPT ===")
required = {"person", "event", "value"}
if not required.issubset(transcript.columns):
    raise ValueError("transcript.csv must contain these columns: person, event, value")

def parse_dict(value):
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError, TypeError):
        return {}

parsed = transcript["value"].apply(parse_dict)
value_df = pd.json_normalize(parsed)
if value_df.shape[1] > 0:
    value_df.columns = ["value_" + str(c) for c in value_df.columns]
    work = pd.concat(
        [transcript.drop(columns=["value"]).reset_index(drop=True),
         value_df.reset_index(drop=True)], axis=1
    )
else:
    work = transcript.drop(columns=["value"]).copy()

work["person"] = work["person"].astype("string")
work["event"] = work["event"].astype("string").str.lower()
if "time" not in work.columns:
    work["time"] = np.nan
work["time"] = pd.to_numeric(work["time"], errors="coerce")

for column in ["value_amount", "value_reward"]:
    if column not in work.columns:
        work[column] = np.nan

work["is_transaction"] = work["event"].eq("transaction").astype(int)
work["is_offer_received"] = work["event"].eq("offer received").astype(int)
work["is_offer_viewed"] = work["event"].eq("offer viewed").astype(int)
work["is_offer_completed"] = work["event"].eq("offer completed").astype(int)
work["transaction_amount_numeric"] = pd.to_numeric(
    work["value_amount"], errors="coerce"
).fillna(0) * work["is_transaction"]
work["reward_numeric"] = pd.to_numeric(work["value_reward"], errors="coerce").fillna(0)

print("Event types:")
display(work["event"].value_counts().to_frame("count"))

customer_features = work.groupby("person", dropna=False).agg(
    total_events=("event", "size"),
    total_transactions=("is_transaction", "sum"),
    offers_received=("is_offer_received", "sum"),
    offers_viewed=("is_offer_viewed", "sum"),
    offers_completed=("is_offer_completed", "sum"),
    total_transaction_amount=("transaction_amount_numeric", "sum"),
    total_rewards=("reward_numeric", "sum"),
    first_event_time=("time", "min"),
    last_event_time=("time", "max")
).reset_index().rename(columns={"person": "customer_id"})

customer_features["customer_id"] = customer_features["customer_id"].astype("string")
customer_features["target_offer_completed"] = (
    customer_features["offers_completed"] > 0
).astype(int)

if "id" not in profile.columns:
    raise ValueError("profile.csv must contain an 'id' column.")
profile_work = profile.copy()
profile_work["id"] = profile_work["id"].astype("string")
profile_work = profile_work.rename(columns={"id": "customer_id"})
data = profile_work.merge(customer_features, on="customer_id", how="inner")

if data.empty:
    raise ValueError("No matching customer IDs found between profile.csv and transcript.csv.")

print("\\nMerged dataset shape:", data.shape)
print("Target distribution:")
display(data["target_offer_completed"].value_counts().to_frame("customers"))

print("\\n=== TRAIN DECISION TREE ===")
# Remove customer ID, target and direct outcome count from the predictors.
# Note: features use the full transcript observation period, so this is a retrospective
# classification/EDA exercise, not a real-time prediction of future offer completion.
drop_cols = [
    c for c in ["customer_id", "target_offer_completed", "offers_completed", "became_member_on"]
    if c in data.columns
]
X = data.drop(columns=drop_cols)
y = data["target_offer_completed"].astype(int)

# Drop entirely missing and constant columns.
bad_cols = [c for c in X.columns if X[c].isna().all() or X[c].nunique(dropna=True) <= 1]
X = X.drop(columns=bad_cols)

if X.shape[1] == 0:
    raise ValueError("No usable feature columns remain.")
if y.nunique() < 2:
    raise ValueError("The target contains only one class, so Decision Tree cannot be trained.")

numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols)
])

stratify_arg = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=stratify_arg
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("tree", DecisionTreeClassifier(
        max_depth=5, min_samples_split=20, min_samples_leaf=10,
        class_weight="balanced", random_state=RANDOM_STATE
    ))
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\\n=== MODEL PERFORMANCE ===")
print("Accuracy :", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, predictions, zero_division=0), 4))
print("F1-score :", round(f1_score(y_test, predictions, zero_division=0), 4))
print("\\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
ConfusionMatrixDisplay(
    confusion_matrix=matrix, display_labels=["No completion", "Completed"]
).plot(values_format="d")
plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.show()

print("\\n=== DECISION TREE VISUALIZATION ===")
fitted_preprocessor = model.named_steps["preprocessor"]
tree_model = model.named_steps["tree"]
try:
    feature_names = fitted_preprocessor.get_feature_names_out()
except Exception:
    feature_names = ["feature_" + str(i) for i in range(tree_model.n_features_in_)]

plt.figure(figsize=(24, 12))
plot_tree(
    tree_model, feature_names=list(feature_names),
    class_names=["No completion", "Completed"], filled=False,
    rounded=True, max_depth=4, fontsize=8
)
plt.title("Decision Tree (shown through depth 4)")
plt.tight_layout()
plt.show()

print("\\n=== FEATURE IMPORTANCE ===")
importance = pd.DataFrame({
    "feature": list(feature_names),
    "importance": tree_model.feature_importances_
}).sort_values("importance", ascending=False)
display(importance.head(20))

if not importance.empty:
    top = importance.head(15).sort_values("importance")
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top Decision Tree Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

print("\\n=== SAVE OUTPUTS ===")
output_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
os.makedirs(output_dir, exist_ok=True)
model_file = os.path.join(output_dir, "starbucks_decision_tree_model.joblib")
metrics_file = os.path.join(output_dir, "decision_tree_metrics.csv")
importance_file = os.path.join(output_dir, "decision_tree_feature_importance.csv")

joblib.dump(model, model_file)
pd.DataFrame([{
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(y_test, predictions, zero_division=0),
    "recall": recall_score(y_test, predictions, zero_division=0),
    "f1_score": f1_score(y_test, predictions, zero_division=0),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "feature_count": X.shape[1]
}]).to_csv(metrics_file, index=False)
importance.to_csv(importance_file, index=False)

print("Saved model:", model_file)
print("Saved metrics:", metrics_file)
print("Saved feature importance:", importance_file)
print("EDA and Decision Tree complete.")
