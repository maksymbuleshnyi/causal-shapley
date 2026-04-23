
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler


HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "german_credit", "german_credit_data.csv")

df = pd.read_csv(CSV_PATH, index_col=0)
df = df[["Age", "Sex", "Saving accounts", "Credit amount", "Risk"]].copy()
df["Saving accounts"] = df["Saving accounts"].fillna("unknown")

le_sex = LabelEncoder()
le_saving = LabelEncoder()
le_risk = LabelEncoder()
df["Sex_encoded"] = le_sex.fit_transform(df["Sex"])
df["Saving_accounts_encoded"] = le_saving.fit_transform(df["Saving accounts"])
df["Risk_encoded"] = le_risk.fit_transform(df["Risk"])

feature_cols = ["Age", "Sex_encoded", "Saving_accounts_encoded", "Credit amount"]
X = df[feature_cols].values
y = df["Risk_encoded"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    alpha=0.001,
    learning_rate="adaptive",
    max_iter=1000,
    random_state=42,
)
mlp.fit(X_train_scaled, y_train)


def model_predict(X_mat: np.ndarray) -> np.ndarray:
    """Model probability of HIGH RISK."""
    return mlp.predict_proba(X_mat)[:, 1]


AGE = 0
SEX = 1
SAVINGS = 2
AMOUNT = 3

AGE_NO_T = 0
SAVINGS_NO_T = 1
AMOUNT_NO_T = 2

T_FEMALE = scaler.transform([[0, 1, 0, 0]])[0][SEX]
T_MALE = scaler.transform([[0, 0, 0, 0]])[0][SEX]


y_hat = model_predict(X_test_scaled)
T_test = X_test_scaled[:, SEX]

mask_women = T_test == T_FEMALE
mask_men = T_test == T_MALE

X_no_T = np.delete(X_test_scaled, SEX, axis=1)
X_women = X_no_T[mask_women]
X_men = X_no_T[mask_men]

mu1 = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mu1.fit(X_women, y_hat[mask_women])

mu0 = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mu0.fit(X_men, y_hat[mask_men])


case_woman = {"Age": 34, "Sex": 1, "Saving accounts": 1, "Credit amount": 1569}
features_woman = scaler.transform(
    [[
        case_woman["Age"],
        case_woman["Sex"],
        case_woman["Saving accounts"],
        case_woman["Credit amount"],
    ]]
)
sample = features_woman[0]


def ate_g_computation() -> float:
    mu1_mean = mu1.predict(X_women).mean()
    mu0_mean = mu0.predict(X_men).mean()
    return float(mu1_mean - mu0_mean)


ate = ate_g_computation()

def cde_pinned(pin_col_no_T: int, pin_value: float) -> float:
    Xw = X_women.copy()
    Xw[:, pin_col_no_T] = pin_value
    Xm = X_men.copy()
    Xm[:, pin_col_no_T] = pin_value
    return float(mu1.predict(Xw).mean() - mu0.predict(Xm).mean())


cde_savings = cde_pinned(SAVINGS_NO_T, sample[SAVINGS])
cde_amount = cde_pinned(AMOUNT_NO_T, sample[AMOUNT])

pe_savings = ate - cde_savings
pe_amount = ate - cde_amount

def filter_close(col_idx: int, value: float, tol: float = 0.2) -> np.ndarray:
    return np.abs(X_test_scaled[:, col_idx] - value) < tol


def conditional_cate(mask: np.ndarray) -> float:
    Xf = X_test_scaled[mask]
    Tf = Xf[:, SEX]
    Xf_w = np.delete(Xf[Tf == T_FEMALE], SEX, axis=1)
    Xf_m = np.delete(Xf[Tf == T_MALE], SEX, axis=1)
    if len(Xf_w) == 0 or len(Xf_m) == 0:
        return float("nan")
    return float(mu1.predict(Xf_w).mean() - mu0.predict(Xf_m).mean())


m_sav = filter_close(SAVINGS, sample[SAVINGS])
m_amt = filter_close(AMOUNT, sample[AMOUNT])
m_age = filter_close(AGE, sample[AGE])

cate_sav_age = conditional_cate(m_sav & m_age)
cate_amt_age = conditional_cate(m_amt & m_age)
cate_sav_amt_age = conditional_cate(m_sav & m_amt & m_age)

pw_amount = cate_sav_amt_age - cate_sav_age
pw_savings = cate_sav_amt_age - cate_amt_age


print("=" * 60)
print("German Credit: PE-SHAP (corrected) and PW-SHAP")
print("=" * 60)
print(f"N_test: {X_test_scaled.shape[0]}  "
      f"(women: {mask_women.sum()}, men: {mask_men.sum()})")
print(f"Sample (scaled):     {sample}")
print()
print(f"ATE (g-computation):           {ate:+.4f}")
print(f"CDE(X_S pinned, g-computation): {cde_savings:+.4f}")
print(f"CDE(X_A pinned, g-computation): {cde_amount:+.4f}")
print()
print("PE-SHAP (correct, backdoor adjustment over Age):")
print(f"  lambda_{{X_S}} (Saving accounts):  {pe_savings:+.4f}")
print(f"  lambda_{{X_A}} (Credit amount):    {pe_amount:+.4f}")
print()
print("PW-SHAP (conditional CATE difference):")
print(f"  psi_{{T -> X_S -> Y}} (Saving accounts): {pw_savings:+.4f}")
print(f"  psi_{{T -> X_A -> Y}} (Credit amount):   {pw_amount:+.4f}")
print()
print(f"Cond CATE conditioning sizes:  "
      f"|full|={int((m_sav & m_amt & m_age).sum())}, "
      f"|noX_A|={int((m_sav & m_age).sum())}, "
      f"|noX_S|={int((m_amt & m_age).sum())}")
