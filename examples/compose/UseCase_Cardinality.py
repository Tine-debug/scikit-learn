import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Reuse the synthetic X from earlier or build a quick one:
rng = np.random.default_rng(42) # seed 42
X = pd.DataFrame({
    "age": rng.integers(18, 70, size=1200),                                     # number, high cardinality
    "income": rng.normal(50000, 12000, size=1200).round(0),                     # number, high cardinality
    "number_cars": rng.choice([0, 1, 2], size=1200),                            # number, low cardinality
    "city": rng.choice(["Luleå", "Stockholm", "Umeå"], size=1200),              # string, low cardinality
    "segment": rng.choice(["A", "B", "C", "D"], size=1200),                     # string, low cardinality
    "zip_code": [f"97{str(i).zfill(3)}" for i in rng.integers(0, 400, 1200)],   # string, high cardinality. <- ignore this
})

# A rudimentary synthetic target that depends on a mix of numeric & categorical signals
y = (
    (X["income"] > 52000).astype(int)
    ^ (X["city"].isin(["Stockholm"]).astype(int))
    ^ (X["segment"].isin(["B", "D"]).astype(int))
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

CARD_THRESH = 5  # choose what "low vs high" means

num_selector  = make_column_selector(dtype_include=np.number, cardinality = 'high', cardinality_threshold = CARD_THRESH)
low_selector  = make_column_selector(cardinality='low',  cardinality_threshold=CARD_THRESH)

preprocess_card = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_selector),
        ("low_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_selector),
    ]
)

clf = Pipeline([
    ("prep", preprocess_card),
    ("model", RandomForestClassifier(n_estimators=300, random_state=0))
])


clf.fit(X_train, y_train)
# Show first 5 transformed rows
X_card = clf.named_steps["prep"].transform(X_test.head(5))
feat_card = clf.named_steps["prep"].get_feature_names_out()
df_card = pd.DataFrame(X_card, columns=feat_card, index=X_test.head(5).index)

print("\n=== first 5 transformed rows ===")
print(df_card.round(3))

# Group columns by transformer to see treatment
num_cols  = [c for c in feat_card if c.startswith("num__")]
low_cols  = [c for c in feat_card if c.startswith("low_ohe__")]

print("\n[Numeric → StandardScaler] (decimal, mean≈0, std≈1)")
print(df_card[num_cols].round(3).head())

print("\n[Low-cardinality → OneHotEncoder] (0/1 indicators)")
print(df_card[low_cols].astype(int).head())  # OHE yields 0/1

pred = clf.predict(X_test)
acc_card = accuracy_score(y_test, pred)
n_feats_card = clf.named_steps["prep"].get_feature_names_out().shape[0]

print(f"accuracy={acc_card:.3f} | features={n_feats_card}")
print("LOW-card columns:", low_selector(X_train))
