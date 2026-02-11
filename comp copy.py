# =====================================================
# MAX AUC XGBOOST – COMPETITION MODE (NO CV)
# =====================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import xgboost
print("XGBoost version:", xgboost.__version__)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import optuna

# =====================================================
# 1. LOAD DATA
# =====================================================

file_path = r"C:\Personal\Educational\Projects\ICSSR-Tech-1st-Round-Submission\Customer_Churn.xlsx"
df = pd.read_excel(file_path)

df.drop(columns=["customerID"], errors="ignore", inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(0, inplace=True)
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

# =====================================================
# 2. FEATURE ENGINEERING (STRONG SIGNAL FEATURES)
# =====================================================

df["avg_monthly"] = df["TotalCharges"] / (df["tenure"] + 1)
df["tenure_x_charge"] = df["tenure"] * df["MonthlyCharges"]
df["charge_per_service"] = df["MonthlyCharges"] / (df["tenure"] + 1)

df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
df["fiber_flag"] = (df["InternetService"] == "Fiber optic").astype(int)

service_cols = [
    "PhoneService","MultipleLines","OnlineSecurity",
    "OnlineBackup","DeviceProtection",
    "TechSupport","StreamingTV","StreamingMovies"
]

existing_services = [c for c in service_cols if c in df.columns]
df["service_count"] = df[existing_services].apply(
    lambda row: sum(row == "Yes"), axis=1
)

# =====================================================
# 3. PREPARE DATA (90/10 SPLIT)
# =====================================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.1,     # 90/10 split
    stratify=y,
    random_state=42
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# =====================================================
# 4. OPTUNA — DIRECT AUC OPTIMIZATION
# =====================================================

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 4000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.1),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 15),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 50),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50),
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "predictor": "gpu_predictor",
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1
    }

    model = XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        
    )

    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)


print("🔥 Running Aggressive AUC Search...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

best_params = study.best_params

best_params.update({
    "scale_pos_weight": scale_pos_weight,
    "tree_method": "hist",
    "predictor": "gpu_predictor",
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1
})

print("\n🏆 BEST PARAMS:")
print(best_params)

# =====================================================
# 5. FINAL TRAINING
# =====================================================

final_model = XGBClassifier(
    **best_params,
    
)

final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

val_probs = final_model.predict_proba(X_val)[:,1]
final_auc = roc_auc_score(y_val, val_probs)

print("\n🚀 FINAL VALIDATION AUC:", final_auc)
