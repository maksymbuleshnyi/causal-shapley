"""
Experiment 2 — Parallel mediators with interaction.

DGP (no confounder):

    T  ~ Bernoulli(0.5)
    M1 = alpha1 * T + U1
    M2 = alpha2 * T + U2
    Y  = beta1*T + beta2*M1 + beta3*M2 + beta4*M1*T + beta5*M1*M2 + UY

Ground truth path effects (averaged over control-arm samples):

    PE-SHAP(T -> M1 -> Y)   =  alpha1 * (beta2 + beta4 + alpha2 * beta5)
    PW-SHAP(T -> M1 -> Y)   = -alpha1 * (beta2 + beta4)
    PE-SHAP(T -> M2 -> Y)   =  alpha2 * (beta3 + alpha1 * beta5)
    PW-SHAP(T -> M2 -> Y)   = -alpha2 * (beta3)

Usage:
    python -m experiments.experiment2_parallel
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulations.simulations import (
    parallel_interaction_dataset,
    parallel_ground_truth,
    PAR_T_COL,
    PAR_M1_COL,
    PAR_M2_COL,
)
from experiments._common import train_model, split_train_eval
from path_wise.synthetic_attribution import (
    attribution_average,
    attribution_summary,
)

DEFAULTS = dict(
    alpha1=0.5, alpha2=0.7,
    beta1=0.3, beta2=0.2, beta3=-1.1,
    beta4=0.2, beta5=0.8,
    sigma=0.5,
)


def run_once(n_total=6000, n_eval=1000, seed=0, model_type="mlp", **params):
    params = {**DEFAULTS, **params}
    X, y = parallel_interaction_dataset(n=n_total, seed=seed, **params)
    X_train, y_train, X_eval, y_eval = split_train_eval(
        X, y, eval_size=n_eval, seed=seed,
    )
    model = train_model(X_train, y_train, model_type=model_type)

    # Evaluate on control-arm samples to recover analytic formulas.
    control_mask = X_eval[:, PAR_T_COL] == 0
    X_eval_control = X_eval[control_mask]

    per_sample = attribution_average(
        model, X_train, T_col=PAR_T_COL,
        mediator_cols=[PAR_M1_COL, PAR_M2_COL],
        X_eval=X_eval_control,
        base_cond_cols=(),
        eps_mediator=0.2,
    )
    summary = attribution_summary(per_sample)
    gt = parallel_ground_truth(**params)
    return summary, gt, params


def parallel_table(summary, gt):
    """Headline table for the T -> M1 -> Y path."""
    rows = []
    rows.append({
        "Method": "PE-SHAP",
        "T -> M1 -> Y (mean)": summary[f"pe_m{PAR_M1_COL}"]["mean"],
        "std": summary[f"pe_m{PAR_M1_COL}"]["std"],
        "GT": gt["path_t_m1_y"],
    })
    rows.append({
        "Method": "PW-SHAP",
        "T -> M1 -> Y (mean)": summary[f"pw_m{PAR_M1_COL}"]["mean"],
        "std": summary[f"pw_m{PAR_M1_COL}"]["std"],
        "GT_pw": None,
    })
    rows.append({
        "Method": "Causal indirect (lumped)",
        "T -> M1 -> Y (mean)": summary["causal_indirect"]["mean"],
        "std": summary["causal_indirect"]["std"],
        "GT": gt["path_t_m1_y"] + gt["path_t_m2_y"] - gt["interaction_surplus"],
    })
    return pd.DataFrame(rows)


def beta5_sweep(beta5_grid=(0.0, 0.25, 0.5, 1.0, 2.0), **kwargs):
    records = []
    for b5 in beta5_grid:
        summary, gt, params = run_once(beta5=b5, **kwargs)
        records.append({
            "beta5": b5,
            "PE-SHAP (M1)": summary[f"pe_m{PAR_M1_COL}"]["mean"],
            "PW-SHAP (M1)": summary[f"pw_m{PAR_M1_COL}"]["mean"],
            "Causal indirect": summary["causal_indirect"]["mean"],
            "GT path M1": gt["path_t_m1_y"],
            "GT PW anchor (-a1(b2+b4))":
                -params["alpha1"] * (params["beta2"] + params["beta4"]),
        })
    return pd.DataFrame(records)


def plot_beta5_sweep(df, path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["beta5"], df["GT path M1"], "--", color="black",
            label=r"GT $\alpha_1(\beta_2+\beta_4+\alpha_2\beta_5)$")
    ax.plot(df["beta5"], df["PE-SHAP (M1)"], "o-", label="PE-SHAP")
    ax.plot(df["beta5"], df["PW-SHAP (M1)"], "s-", label="PW-SHAP")
    ax.plot(df["beta5"], df["GT PW anchor (-a1(b2+b4))"], ":", color="gray",
            label=r"PW anchor $-\alpha_1(\beta_2+\beta_4)$")
    ax.set_xlabel(r"$\beta_5$ (interaction strength)")
    ax.set_ylabel(r"Attribution for $T\to M_1\to \hat Y$")
    ax.set_title(r"Experiment 2: path attribution vs. $\beta_5$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def sign_flip_sweep(beta5_grid=None, **kwargs):
    """
    Choose parameters so that the sign of
    ``alpha1*(beta2+beta4+alpha2*beta5)`` flips as beta5 varies.
    Easy recipe: set (beta2+beta4) negative and alpha2*beta5 positive.
    """
    if beta5_grid is None:
        beta5_grid = np.linspace(-1.0, 2.0, 13)
    base = dict(alpha1=0.7, alpha2=0.5,
                beta1=0.3, beta2=-0.6, beta3=0.4,
                beta4=0.1, sigma=0.5)
    records = []
    for b5 in beta5_grid:
        summary, gt, params = run_once(beta5=float(b5), **{**base, **kwargs})
        records.append({
            "beta5": b5,
            "PE-SHAP (M1)": summary[f"pe_m{PAR_M1_COL}"]["mean"],
            "PW-SHAP (M1)": summary[f"pw_m{PAR_M1_COL}"]["mean"],
            "GT path M1": gt["path_t_m1_y"],
            "PW anchor":
                -params["alpha1"] * (params["beta2"] + params["beta4"]),
        })
    return pd.DataFrame(records)


def plot_sign_flip(df, path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(df["beta5"], df["GT path M1"], "--", color="black",
            label="Analytic GT (crosses zero)")
    ax.plot(df["beta5"], df["PE-SHAP (M1)"], "o-", label="PE-SHAP (tracks GT)")
    ax.plot(df["beta5"], df["PW-SHAP (M1)"], "s-",
            label="PW-SHAP (stuck on wrong side)")
    ax.set_xlabel(r"$\beta_5$")
    ax.set_ylabel(r"Attribution for $T\to M_1\to \hat Y$")
    ax.set_title("Experiment 2: sign-flip demonstration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


if __name__ == "__main__":
    print("=== Experiment 2: Parallel mediators (default config) ===")
    summary, gt, params = run_once(beta5=0.5)
    print(parallel_table(summary, gt).to_string(index=False))

    print("\n=== Experiment 2: beta5 sweep ===")
    df_sweep = beta5_sweep()
    print(df_sweep.to_string(index=False))
    plot_beta5_sweep(df_sweep, "experiments/experiment2_beta5_sweep.png")

    print("\n=== Experiment 2: sign-flip sweep ===")
    df_flip = sign_flip_sweep()
    print(df_flip.to_string(index=False))
    plot_sign_flip(df_flip, "experiments/experiment2_sign_flip.png")
    print("Saved figures to experiments/experiment2_*.png")
