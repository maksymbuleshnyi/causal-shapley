from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler


HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "..", "data", "german_credit.csv")


# Feature column indices (after one-hot of Sex / Savings)
AGE = 0
SEX = 1
SAVINGS = 2
AMOUNT = 3

AGE_NO_T = 0
SAVINGS_NO_T = 1
AMOUNT_NO_T = 2


def _load_data():
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
    return X, y


def _filter_close(X_test_scaled, col_idx, value, tol=0.2):
    return np.abs(X_test_scaled[:, col_idx] - value) < tol


def _conditional_cate(X_test_scaled, mu1, mu0, T_FEMALE, T_MALE, mask):
    Xf = X_test_scaled[mask]
    Tf = Xf[:, SEX]
    Xf_w = np.delete(Xf[Tf == T_FEMALE], SEX, axis=1)
    Xf_m = np.delete(Xf[Tf == T_MALE], SEX, axis=1)
    if len(Xf_w) == 0 or len(Xf_m) == 0:
        return float("nan")
    return float(mu1.predict(Xf_w).mean() - mu0.predict(Xf_m).mean())


def run_one(X, y, seed):
    """One MC iteration: split, train, compute attributions for the woman case."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y,
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
        random_state=seed,
    )
    mlp.fit(X_train_scaled, y_train)

    def model_predict(X_mat):
        return mlp.predict_proba(X_mat)[:, 1]

    T_FEMALE = scaler.transform([[0, 1, 0, 0]])[0][SEX]
    T_MALE = scaler.transform([[0, 0, 0, 0]])[0][SEX]

    y_hat = model_predict(X_test_scaled)
    T_test = X_test_scaled[:, SEX]
    mask_women = T_test == T_FEMALE
    mask_men = T_test == T_MALE

    X_no_T = np.delete(X_test_scaled, SEX, axis=1)
    X_women = X_no_T[mask_women]
    X_men = X_no_T[mask_men]

    mu1 = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=seed)
    mu1.fit(X_women, y_hat[mask_women])
    mu0 = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=seed)
    mu0.fit(X_men, y_hat[mask_men])

    case_woman = {"Age": 34, "Sex": 1, "Saving accounts": 1, "Credit amount": 1569}
    sample = scaler.transform([[
        case_woman["Age"], case_woman["Sex"],
        case_woman["Saving accounts"], case_woman["Credit amount"],
    ]])[0]

    # ATE via g-computation
    ate = float(mu1.predict(X_women).mean() - mu0.predict(X_men).mean())

    def cde_pinned(pin_col_no_T, pin_value):
        Xw = X_women.copy(); Xw[:, pin_col_no_T] = pin_value
        Xm = X_men.copy();   Xm[:, pin_col_no_T] = pin_value
        return float(mu1.predict(Xw).mean() - mu0.predict(Xm).mean())

    cde_savings = cde_pinned(SAVINGS_NO_T, sample[SAVINGS])
    cde_amount = cde_pinned(AMOUNT_NO_T, sample[AMOUNT])
    pe_savings = ate - cde_savings
    pe_amount = ate - cde_amount

    # PW-SHAP via conditional CATE differences
    m_sav = _filter_close(X_test_scaled, SAVINGS, sample[SAVINGS])
    m_amt = _filter_close(X_test_scaled, AMOUNT, sample[AMOUNT])
    m_age = _filter_close(X_test_scaled, AGE, sample[AGE])

    cate_sav_age = _conditional_cate(X_test_scaled, mu1, mu0, T_FEMALE, T_MALE,
                                     m_sav & m_age)
    cate_amt_age = _conditional_cate(X_test_scaled, mu1, mu0, T_FEMALE, T_MALE,
                                     m_amt & m_age)
    cate_sav_amt_age = _conditional_cate(X_test_scaled, mu1, mu0,
                                         T_FEMALE, T_MALE,
                                         m_sav & m_amt & m_age)
    pw_amount = cate_sav_amt_age - cate_sav_age
    pw_savings = cate_sav_amt_age - cate_amt_age

    # Population-level NIE / NDE
    nde = float(mu1.predict(X_men).mean() - mu0.predict(X_men).mean())
    nie = float(mu1.predict(X_women).mean() - mu1.predict(X_men).mean())

    # Population-level PE-SHAP
    pop_pe_s = []
    pop_pe_a = []
    for x in X_test_scaled:
        pop_pe_s.append(ate - cde_pinned(SAVINGS_NO_T, x[SAVINGS]))
        pop_pe_a.append(ate - cde_pinned(AMOUNT_NO_T, x[AMOUNT]))
    pop_pe_savings = float(np.mean(pop_pe_s))
    pop_pe_amount = float(np.mean(pop_pe_a))

    return {
        "ate": ate,
        "cde_savings_local": cde_savings,
        "cde_amount_local": cde_amount,
        "pe_savings_local": pe_savings,
        "pe_amount_local": pe_amount,
        "pw_savings_local": pw_savings,
        "pw_amount_local": pw_amount,
        "nde": nde,
        "nie": nie,
        "pe_savings_pop": pop_pe_savings,
        "pe_amount_pop": pop_pe_amount,
    }


def aggregate(records):
    keys = list(records[0].keys())
    return {k: (float(np.mean([r[k] for r in records])),
                float(np.std([r[k] for r in records])))
            for k in keys}


def _block(title, rows, width=72):
    lines = ['=' * width, title, '=' * width,
             f"{'Metric':<35}{'Mean':>14}{'Std':>14}",
             '-' * width]
    for name, (m, s) in rows:
        lines.append(f"{name:<35}{m:>+14.4f}{s:>14.4f}")
    return '\n'.join(lines)


def build_summary(agg, n_runs):
    parts = []
    parts.append('=' * 80)
    parts.append("German Credit -- Sex attribution")
    parts.append('=' * 80)
    parts.append(f"n_runs = {n_runs}   "
                 f"counterfactual woman: Age=34, Saving=1, Amount=1569")
    parts.append("MC variation comes from train/test split and MLP seed.")
    parts.append('')

    parts.append(_block("Population-level effects", [
        ("ATE (g-computation)",  agg["ate"]),
        ("NDE",                  agg["nde"]),
        ("NIE",                  agg["nie"]),
    ]))
    parts.append('')

    parts.append(_block("PE-SHAP at counterfactual sample", [
        ("CDE(Saving=sample, pinned)", agg["cde_savings_local"]),
        ("CDE(Amount=sample, pinned)", agg["cde_amount_local"]),
        ("lambda_{X_S} (Saving)",      agg["pe_savings_local"]),
        ("lambda_{X_A} (Amount)",      agg["pe_amount_local"]),
    ]))
    parts.append('')

    parts.append(_block("PW-SHAP at counterfactual sample", [
        ("Psi_{T -> X_S -> Y} (Saving)", agg["pw_savings_local"]),
        ("Psi_{T -> X_A -> Y} (Amount)", agg["pw_amount_local"]),
    ]))
    parts.append('')

    parts.append(_block("Population-level PE-SHAP (averaged over test set)", [
        ("E[lambda_{X_S}]", agg["pe_savings_pop"]),
        ("E[lambda_{X_A}]", agg["pe_amount_pop"]),
    ]))
    parts.append('')

    # Combined row layout: components of the Sex attribution.
    cell_w = 22
    col_w = 14
    total_w = col_w + 4 * cell_w
    parts.append('=' * total_w)
    parts.append("Sex attribution components")
    parts.append('=' * total_w)
    header = (f"{'Method':<{col_w}}"
              f"{'X_A path':^{cell_w}}{'X_S path':^{cell_w}}"
              f"{'NDE':^{cell_w}}{'NIE':^{cell_w}}")
    parts.append(header)
    parts.append('-' * total_w)

    def cell(pair):
        m, s = pair
        return f"{m:+.3f} +/- {s:.3f}"

    parts.append(
        f"{'PW-SHAP':<{col_w}}"
        f"{cell(agg['pw_amount_local']):^{cell_w}}"
        f"{cell(agg['pw_savings_local']):^{cell_w}}"
        f"{'---':^{cell_w}}{'---':^{cell_w}}"
    )
    parts.append(
        f"{'PE-SHAP':<{col_w}}"
        f"{cell(agg['pe_amount_local']):^{cell_w}}"
        f"{cell(agg['pe_savings_local']):^{cell_w}}"
        f"{'---':^{cell_w}}{'---':^{cell_w}}"
    )
    parts.append(
        f"{'g-formula':<{col_w}}"
        f"{'---':^{cell_w}}{'---':^{cell_w}}"
        f"{cell(agg['nde']):^{cell_w}}{cell(agg['nie']):^{cell_w}}"
    )
    parts.append('=' * total_w)

    return '\n'.join(parts)


def main(n_runs=5):
    X, y = _load_data()
    records = []
    for r in range(n_runs):
        print(f"  run {r+1}/{n_runs} ...", end=" ", flush=True)
        records.append(run_one(X, y, seed=r))
        print("done")

    agg = aggregate(records)

    summary = build_summary(agg, n_runs)
    print('\n' + summary)

    os.makedirs('./results', exist_ok=True)
    out_path = './results/german_credit.txt'
    with open(out_path, 'w') as f:
        f.write(summary + '\n')
    print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    main()
